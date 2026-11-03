# tests/_phlop/app/pfm/test_pfm_stat.py
#
# needs neither libpfm nor perf counter access: feeds synthetic perf json

import unittest

from phlop.app import perf
from phlop.app.pfm import stat
from phlop.app.pfm.check_events import _parse_detected_pmus, _parse_encodings
from phlop.app.pfm.sets import SETS, generic

ZEN5 = {s.name: s for s in SETS["amd64_fam1ah_zen5"]}
N2 = {s.name: s for s in SETS["arm_n2"]}


def rows(**counts):
    return [
        {"counter-value": f"{v:.6f}", "event": k, "pcnt-running": 100.0}
        for k, v in counts.items()
    ]


def evaluate(eset, rws, cpu_flags=(), **ctx):
    res = stat.SetResult(eset.name, eset.desc, [])
    reported = [(a, a, "") for a in eset.events]
    res.events = stat.collect(eset, rws, reported)
    stat.evaluate(eset, res, {"cpu_flags": frozenset(cpu_flags), **ctx})
    return res


class PfmStatTest(unittest.TestCase):
    def test_topdown_sums_to_one(self):
        res = evaluate(
            ZEN5["topdown"],
            rows(cycles=1000, ops=3200, fe=1600, be=2400, smt=400, dispatched=3600),
        )
        m = res.metrics
        self.assertAlmostEqual(m["retiring"], 0.4)
        self.assertAlmostEqual(m["frontend_bound"], 0.2)
        self.assertAlmostEqual(m["backend_bound"], 0.3)
        self.assertAlmostEqual(m["bad_speculation"], 0.05)
        self.assertAlmostEqual(m["smt_contention"], 0.05)
        self.assertAlmostEqual(sum(m.values()), 1.0)

    def test_vector_hints(self):
        res = evaluate(
            ZEN5["vector"],
            rows(x87=0, scalar=900, p128=0, p256=100, p512=0, sse_avx_faults=3),
            cpu_flags=["avx512f"],
        )
        self.assertAlmostEqual(res.metrics["packed_uops"], 0.1)
        hints = " ".join(res.hints)
        self.assertIn("many scalar FP uops", hints)
        self.assertIn("AVX-512", hints)
        self.assertIn("SSE/AVX", hints)

    def test_backend_hint_load_queue(self):
        counts = {
            "cycles": 1000,
            "be": 6000,
            "not_complete": 1000,
            "load_not_complete": 100,
        }
        lq = evaluate(ZEN5["backend"], rows(**counts, lq_stall=600, fp_sched_stall=0))
        dep = evaluate(ZEN5["backend"], rows(**counts, lq_stall=10, fp_sched_stall=0))
        self.assertIn("load queue", " ".join(lq.hints))
        self.assertIn("dependency chains", " ".join(dep.hints))

    def test_light(self):
        res = evaluate(ZEN5["light"], rows(cycles=100, instructions=300, fetches=150))
        self.assertAlmostEqual(res.metrics["ipc"], 3.0)
        self.assertAlmostEqual(res.metrics["instructions_per_fetch"], 2.0)

    def test_zero_fetches_label(self):
        res = evaluate(ZEN5["light"], rows(cycles=100, instructions=300, fetches=0))
        self.assertIsNone(res.metrics["instructions_per_fetch"])
        self.assertEqual(res.labels["instructions_per_fetch"], "none (no RAM fetches)")

    def test_missing_and_multiplexed(self):
        rws = rows(cycles=10, instructions=20, ops=20, branches=5)
        rws[0]["pcnt-running"] = 50.0
        rws.append({"counter-value": "<not counted>", "event": "br_misp"})
        res = evaluate(ZEN5["ipc"], rws)
        self.assertAlmostEqual(res.metrics["ipc"], 2.0)
        self.assertIsNone(res.metrics["branch_mispredict_rate"])
        self.assertEqual(res.events["cycles"].pcnt_running, 50.0)

    def test_generic_symbolic_names(self):
        eset = generic[0]
        rws = rows(**{"cycles:u": 100, "instructions:u": 50})
        reported = [(a, s[5:], "") for a, s in eset.events.items()]
        res = stat.SetResult(eset.name, eset.desc, [])
        res.events = stat.collect(eset, rws, reported)
        stat.evaluate(eset, res, {})
        self.assertAlmostEqual(res.metrics["ipc"], 0.5)

    def test_counter_budget(self):
        self.assertEqual(ZEN5["flops"].counters_needed(), 5)  # 2 pair events
        self.assertTrue(all(s.counters_needed() <= 6 for s in ZEN5.values()))

    def test_parse_check_events(self):
        lines = [
            "Detected PMU models:",
            '\t[51, perf, "perf_events generic PMU"]',
            '\t[676, amd64_fam1ah_zen5, "AMD64 Fam1Ah Zen5"]',
            "Total events: 1 available, 1 supported",
            "Requested Event: RETIRED_OPS",
            "Actual    Event: amd64_fam1ah_zen5::RETIRED_OPS:k=1:u=1",
            "PMU            : AMD64 Fam1Ah Zen5",
            "Codes          : 0x5300c1",
        ]
        self.assertEqual(_parse_detected_pmus(lines), ["perf", "amd64_fam1ah_zen5"])
        self.assertEqual(_parse_encodings(lines), {"RETIRED_OPS": 0x5300C1})

    def test_pmu_terms_config_uses_format_bits(self):
        from unittest import mock

        from phlop.app.pfm import check_events as ce

        amd = {"event": [(0, 7), (32, 35)], "umask": [(8, 15)], "cmask": [(24, 31)]}
        with mock.patch.object(ce, "_format_fields", return_value=amd):
            self.assertEqual(
                ce.pmu_terms_config("cpu/event=0x165,umask=0xc8/"), 0x10000C865
            )
            with self.assertRaises(ValueError):
                ce.pmu_terms_config("cpu/event=0x1165/")  # 13 bits into 12
            with self.assertRaises(ValueError):
                ce.pmu_terms_config("cpu/period=1/")

    def test_scope_env_format(self):
        from unittest import mock

        light = ZEN5["light"]
        with (
            mock.patch.object(
                stat.ce,
                "encode",
                return_value={
                    "CYCLES_NOT_IN_HALT": 0x530076,
                    "RETIRED_INSTRUCTIONS": 0x5300C0,
                },
            ),
            mock.patch.object(stat.ce, "perf_config_mask", return_value=0xFFF84FFFF),
            mock.patch.object(stat.ce, "pmu_terms_config", return_value=0x10000C865),
        ):
            env = stat.scope_env("light", {"light": light}, "amd64_fam1ah_zen5")
        self.assertEqual(
            env,
            "amd64_fam1ah_zen5/light;cycles=0x76,instructions=0xc0,fetches=0x10000c865",
        )

    def test_scope_env_generic_is_perf_hardware(self):
        sets = {s.name: s for s in generic}
        env = stat.scope_env("light", sets, "generic")
        self.assertEqual(
            env,
            "generic/light;cycles=hw:0x0,instructions=hw:0x1,branches=hw:0x4,"
            "br_misp=hw:0x5",
        )
        self.assertIn("cache_misses=hw:0x3", stat.scope_env("ipc", sets, "generic"))

    def test_pmu_warnings(self):
        from phlop.app.pfm.sets import pmu_warnings

        self.assertEqual(pmu_warnings("amd64_fam1ah_zen5", ["amd64_fam1ah_zen5"]), [])

        (w,) = pmu_warnings("generic", ["amd64_fam19h_zen4"])  # unknown CPU
        self.assertIn("no phlop event sets for PMU amd64_fam19h_zen4", w)
        self.assertIn("potentially inaccurate", w)
        self.assertIn("LIBPFM_FORCE_PMU=amd64_fam19h_zen4", w)  # how to add sets

        (w,) = pmu_warnings("generic", [])  # libpfm sees nothing
        self.assertIn("./sh/setup_pfm.sh", w)

        (w,) = pmu_warnings("generic", [], forced="arm_n2")  # not in this build
        self.assertIn("LIBPFM_FORCE_PMU=arm_n2 is not a PMU model", w)

        # forced to a model with sets, on another CPU: encodings are not this CPU's
        (w,) = pmu_warnings("arm_n2", ["arm_n2"], "arm_n2", ["arm_v2"])
        self.assertIn("this CPU is arm_v2", w)
        self.assertIn("potentially wrong", w)
        self.assertIn("unset LIBPFM_FORCE_PMU", w)
        self.assertEqual(pmu_warnings("arm_n2", ["arm_n2"], "arm_n2", ["arm_n2"]), [])

        # this CPU's model could not be checked: say so, not a mismatch
        (w,) = pmu_warnings("arm_n2", ["arm_n2"], "arm_n2", None)
        self.assertIn("could not check this CPU's own model", w)
        self.assertNotIn("libpfm knows no model", w)

    def test_env_budget_counts_each_level_once(self):
        import io
        from contextlib import redirect_stderr
        from unittest import mock

        with (
            mock.patch.object(stat.ce, "detected_pmus", return_value=[]),
            mock.patch.object(stat.ce, "pmu_warnings", return_value=[]),
            mock.patch.object(stat.ce, "forced_encodings", return_value=False),
            mock.patch.object(stat, "available_counters", return_value=4),
        ):
            err = io.StringIO()
            with redirect_stderr(err):  # ipc's 6 events, replaced by light's 4
                lines = stat.scope_env_exports(["perf_light=ipc", "perf_light=light"])
            self.assertNotIn("will not schedule", err.getvalue())
            self.assertEqual(sum("PHLOP_PERF_LIGHT=" in line for line in lines), 1)
            self.assertIn("generic/light;", lines[0])
            with redirect_stderr(err):
                stat.scope_env_exports(["perf_light=ipc"])
            self.assertIn("will not schedule", err.getvalue())

    def test_access_warnings(self):
        ok, warnings = perf.access_warnings()
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(warnings, list)

    def test_access_warnings_denied(self):
        from unittest import mock

        with (
            mock.patch.object(perf, "paranoid_level", return_value=3),
            mock.patch.object(perf, "privileged", return_value=False),
        ):
            ok, warnings = perf.access_warnings()
        self.assertFalse(ok)
        self.assertIn("perf_event_paranoid=3", warnings[0])

    def test_privileged_is_capabilities_not_uid(self):
        from unittest import mock

        def status(caps):
            return mock.mock_open(read_data=f"Name:\tx\nCapEff:\t{caps:016x}\n")

        with mock.patch("os.geteuid", return_value=0):
            with mock.patch("builtins.open", status(0)):  # root in a container
                self.assertFalse(perf.privileged())
            with mock.patch("builtins.open", status(1 << perf.CAP_PERFMON)):
                self.assertTrue(perf.privileged())
            with mock.patch("builtins.open", side_effect=OSError):
                self.assertTrue(perf.privileged())  # unreadable: uid decides
        with (
            mock.patch("os.geteuid", return_value=1000),
            mock.patch("builtins.open", status(1 << perf.CAP_SYS_ADMIN)),
        ):
            self.assertTrue(perf.privileged())

    def test_access_privileged_without_sysctl(self):
        from unittest import mock

        with (
            mock.patch.object(perf, "paranoid_level", return_value=None),
            mock.patch.object(perf, "privileged", return_value=True),
        ):
            ok, _ = perf.access_warnings()
        self.assertTrue(ok)


class ArmTest(unittest.TestCase):
    def test_core_pmu_detection(self):
        import tempfile
        from pathlib import Path

        from phlop.app.pfm import host

        with tempfile.TemporaryDirectory() as d:
            for name, marker in (
                ("armv8_pmuv3_0", "cpus"),
                ("arm_cmn_0", "cpumask"),  # uncore
                ("software", None),
            ):
                (Path(d, name, "format")).mkdir(parents=True)
                if marker:
                    Path(d, name, marker).write_text("0-3")
            self.assertEqual([p.name for p in host.core_pmus(d)], ["armv8_pmuv3_0"])

    def test_topdown(self):
        # 1000 cycles * 5 slots, 1000 stalled slots: 800 frontend, 200 backend
        counts = rows(
            cycles=1000,
            stall=1000,
            stall_fe=800,
            stall_be=200,
            op_spec=4000,
            op_retired=3600,
            br_misp_spec=0,
        )
        m = evaluate(N2["topdown"], counts, slots=5).metrics
        self.assertAlmostEqual(m["retiring"], 0.9 * 0.8)
        self.assertAlmostEqual(m["bad_speculation"], 0.1 * 0.8)
        self.assertAlmostEqual(m["frontend_bound"], 0.16)
        self.assertAlmostEqual(m["backend_bound"], 0.04)
        self.assertAlmostEqual(
            sum(
                m[k]
                for k in (
                    "retiring",
                    "bad_speculation",
                    "frontend_bound",
                    "backend_bound",
                )
            ),
            1.0,
        )

    def test_topdown_n2_erratum_subtracts_cycles(self):
        counts = rows(
            cycles=1000,
            stall=2000,  # 1000 real + one per cycle
            stall_fe=1800,
            stall_be=200,
            op_spec=4000,
            op_retired=3600,
            br_misp_spec=0,
        )
        m = evaluate(
            N2["topdown"], counts, slots=5, stall_slot_counts_cycles=True
        ).metrics
        self.assertAlmostEqual(m["frontend_bound"], 0.16)
        self.assertAlmostEqual(m["retiring"], 0.9 * 0.8)

    def test_topdown_without_slots_is_none(self):
        counts = rows(
            cycles=1,
            stall=1,
            stall_fe=1,
            stall_be=1,
            op_spec=1,
            op_retired=1,
            br_misp_spec=0,
        )
        self.assertIsNone(evaluate(N2["topdown"], counts).metrics["retiring"])

    def test_sve_flops_scale_with_vector_length(self):
        counts = rows(cycles=100, scale_ops=10, fixed_ops=20)
        self.assertAlmostEqual(
            evaluate(N2["flops"], counts, sve_bits=256).metrics["flops_per_cycle"],
            (10 * 2 + 20) / 100,
        )
        self.assertAlmostEqual(  # no SVE info: 128 bit units
            evaluate(N2["flops"], counts).metrics["flops_per_cycle"], 0.3
        )

    def test_cycle_counter_is_free(self):
        self.assertEqual(N2["topdown"].counters_needed(), 6)  # 7 events
        self.assertTrue(all(s.counters_needed() <= 6 for s in N2.values()))

    def test_slots_fallback_when_caps_reads_zero(self):
        from unittest import mock

        from phlop.app.pfm import host

        n2_r0p0 = (0x41, 0, 0xD49, 0)
        with (
            mock.patch.object(host, "pmu_caps", return_value={"slots": "0x0"}),
            mock.patch.object(host, "arm_midr", return_value=n2_r0p0),
        ):
            self.assertEqual(host.dispatch_slots(), 5)
        with mock.patch.object(host, "pmu_caps", return_value={"slots": "0x8"}):
            self.assertEqual(host.dispatch_slots_and_source(), (8, "caps/slots"))
        with (
            mock.patch.object(host, "pmu_caps", return_value={}),
            mock.patch.object(host, "arm_midr", return_value=None),
        ):
            self.assertIsNone(host.dispatch_slots())

    def test_basic_arm_pmus_have_no_topdown(self):
        from phlop.app.pfm.sets import sets_for

        pmu, sets = sets_for(["arm_n1"])
        self.assertEqual(pmu, "arm_n1")
        self.assertNotIn("topdown", sets)
        self.assertIn("light", sets)


if __name__ == "__main__":
    unittest.main()
