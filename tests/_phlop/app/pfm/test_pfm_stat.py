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

    def test_access_warnings(self):
        ok, warnings = perf.access_warnings()
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(warnings, list)


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
