# phlop/app/pfm/sets.py
#
# Named hardware event sets and the metrics derived from them, per PMU model.
#
# Event specs:
#   "RETIRED_OPS"                 libpfm event string, encoded via check_events
#   "raw:cpu/event=0xaa,umask=7/" perf PMU string, for encodings libpfm lacks
#   "perf:cycles"                 perf symbolic event
#
# Each set is kept within the PMU's counter budget where possible, so counts
# are exact rather than multiplexed estimates. Zen5 has 6 core counters,
# RETIRED_SSE_AVX_FLOPS (event 0x003) is a "large increment" event that
# the kernel schedules on a counter pair, so it costs 2.
#
# Zen5 metric formulas follow linux tools/perf/pmu-events/arch/x86/amdzen5

from collections.abc import Callable
from dataclasses import dataclass, field

PAIR_EVENTS = ("RETIRED_SSE_AVX_FLOPS",)  # cost two counters each
FIXED_EVENTS = ("CPU_CYCLES",)  # arm64 dedicated cycle counter, costs none


def ratio(a, b):
    return None if a is None or b is None or b == 0 else a / b


@dataclass
class Metric:
    name: str
    fn: Callable  # (counts: dict alias->float) -> float | None
    unit: str = ""  # "%": fn returns a fraction, shown * 100
    desc: str = ""
    hint: Callable | None = None  # (value, counts, ctx) -> str | None
    # (counts) -> label shown instead of n/a when fn gives None for a known
    # reason, eg "none (no RAM fetches)" for a ratio over zero fetches
    none_means: Callable | None = None


@dataclass
class EventSet:
    name: str
    desc: str
    events: dict  # alias -> spec
    metrics: list = field(default_factory=list)

    def counters_needed(self):
        def cost(spec):
            if spec in FIXED_EVENTS:
                return 0
            return 2 if any(spec.startswith(p) for p in PAIR_EVENTS) else 1

        return sum(cost(spec) for spec in self.events.values())


def _no_fetches(c):
    return "none (no RAM fetches)" if c.get("fetches") == 0 else None


def _hint(pred, msg):
    return lambda v, c, ctx: msg if v is not None and pred(v) else None


# --------------------------------------------------------------------------- #
# AMD Zen5, family 1Ah

ZEN5_SLOTS = 8  # dispatch width
DDCF = "DEMAND_DATA_CACHE_FILLS_FROM_SYSTEM"
# cache lines filled into L2 from RAM (DRAM near/far + extension memory), demand
# and prefetch, counted per thread. Most streaming RAM traffic arrives via the L2
# prefetcher, so L1D fill sources miss it. The L3 miss counter is uncore (whole
# CCX) so unusable per thread. perf: l2_fill_rsp_src, libpfm names this event
# L2_PREFETCH_MISS_L3 with a misleading description, so use the raw encoding
RAM_FETCHES = "raw:cpu/event=0x165,umask=0xc8/"
FLOPS = "RETIRED_SSE_AVX_FLOPS"
BY_WIDTH = "RETIRED_FP_OPS_BY_WIDTH"


def _slots(c):
    return ZEN5_SLOTS * c["cycles"] if c.get("cycles") is not None else None


def _per_kilo(key, per="instructions"):
    return lambda c: ratio(c.get(key), c.get(per) and c[per] / 1000)


def _sum(*keys):
    return lambda c: (
        None if any(c.get(k) is None for k in keys) else sum(c[k] for k in keys)
    )


def _avx512_hint(v, c, ctx):
    if "avx512f" in ctx.get("cpu_flags", ()) and c.get("p512") == 0 and c.get("p256"):
        return (
            "CPU supports AVX-512 but no 512-bit uops retired: try -march=native "
            "and/or -mprefer-vector-width=512 (gcc), check intrinsics/libraries"
        )
    return None


VECTORIZE_ADVICE = (
    "vectorization reports (gcc -fopt-info-vec-missed, "
    "clang -Rpass-missed=loop-vectorize), aliasing (__restrict), "
    "-ffast-math for reductions"
)


def _backend_core_hint(v, c, ctx):
    if v is None or v <= 0.2:
        return None
    if (ratio(c.get("lq_stall"), c.get("cycles")) or 0) > 0.3:
        return (
            "load throughput bound, the load queue is often full: fewer or wider "
            "loads, keep values in registers, align to 64B to avoid split loads "
            "(see -s tlb cacheline_split_loads_pki)"
        )
    return (
        "core bound: long dependency chains or divides, try more independent "
        "accumulators, -ffast-math reassociation"
    )


def _fp_uops(c):
    return _sum("x87", "scalar", "p128", "p256", "p512")(c)


zen5 = [
    EventSet(
        "light",
        "cheap per scope counters: cycles, instructions, RAM fetches",
        {
            "cycles": "CYCLES_NOT_IN_HALT",
            "instructions": "RETIRED_INSTRUCTIONS",
            "fetches": RAM_FETCHES,
        },
        [
            Metric("ipc", lambda c: ratio(c["instructions"], c["cycles"])),
            Metric(
                "instructions_per_fetch",
                lambda c: ratio(c["instructions"], c["fetches"]),
                desc="instructions per 64B line fetched from RAM",
                none_means=_no_fetches,
                hint=_hint(
                    lambda v: v < 100,
                    "under 100 instructions per RAM fetch: memory bound, improve "
                    "reuse (tiling/blocking), see -s cache,backend",
                ),
            ),
            Metric(
                "fetch_bytes_per_cycle",
                lambda c: ratio(c["fetches"] * 64, c["cycles"]),
                desc="RAM bytes read into L1D per core cycle",
            ),
        ],
    ),
    EventSet(
        "ipc",
        "instructions per cycle and branch prediction",
        {
            "cycles": "CYCLES_NOT_IN_HALT",
            "instructions": "RETIRED_INSTRUCTIONS",
            "ops": "RETIRED_OPS",
            "branches": "RETIRED_BRANCH_INSTRUCTIONS",
            "br_misp": "RETIRED_BRANCH_INSTRUCTIONS_MISPREDICTED",
        },
        [
            Metric(
                "ipc",
                lambda c: ratio(c["instructions"], c["cycles"]),
                desc="retired instructions per cycle, Zen5 peak 8",
                hint=_hint(lambda v: v < 1, "low IPC: see -s topdown"),
            ),
            Metric("ops_per_instruction", lambda c: ratio(c["ops"], c["instructions"])),
            Metric(
                "branch_mispredict_rate",
                lambda c: ratio(c["br_misp"], c["branches"]),
                "%",
                hint=_hint(
                    lambda v: v > 0.02,
                    "branch mispredicts > 2%: data dependent branches, "
                    "consider branchless code / sorting / PGO",
                ),
            ),
            Metric("branch_mpki", _per_kilo("br_misp")),
        ],
    ),
    EventSet(
        "topdown",
        "level 1 pipeline utilization, fraction of dispatch slots",
        {
            "cycles": "CYCLES_NOT_IN_HALT",
            "ops": "RETIRED_OPS",
            "fe": "DISPATCH_STALLS_1:FE_NO_OPS",
            "be": "DISPATCH_STALLS_1:BE_STALLS",
            "smt": "DISPATCH_STALLS_1:SMT_CONTENTION",
            # libpfm lacks the loop buffer umask bit (0x4)
            "dispatched": "raw:cpu/event=0xaa,umask=0x7/",
        },
        [
            Metric(
                "retiring",
                lambda c: ratio(c["ops"], _slots(c)),
                "%",
                hint=_hint(
                    lambda v: v > 0.5,
                    "mostly retiring: speedups come from doing less work, "
                    "fewer instructions (vectorize: -s vector,flops), better algorithm",
                ),
            ),
            Metric(
                "frontend_bound",
                lambda c: ratio(c["fe"], _slots(c)),
                "%",
                hint=_hint(lambda v: v > 0.2, "frontend bound: see -s frontend,icache"),
            ),
            Metric(
                "backend_bound",
                lambda c: ratio(c["be"], _slots(c)),
                "%",
                hint=_hint(
                    lambda v: v > 0.3, "backend bound: see -s backend,cache,tlb"
                ),
            ),
            Metric(
                "bad_speculation",
                lambda c: ratio(c["dispatched"] - c["ops"], _slots(c)),
                "%",
                desc="dispatched ops that did not retire",
                hint=_hint(
                    lambda v: v > 0.1, "bad speculation: see branch metrics in -s ipc"
                ),
            ),
            Metric("smt_contention", lambda c: ratio(c["smt"], _slots(c)), "%"),
        ],
    ),
    EventSet(
        "frontend",
        "frontend latency vs bandwidth",
        {
            "cycles": "CYCLES_NOT_IN_HALT",
            "instructions": "RETIRED_INSTRUCTIONS",
            "fe": "DISPATCH_STALLS_1:FE_NO_OPS",
            "fe_lat": "DISPATCH_STALLS_1:FE_NO_OPS:c=8",  # cycles with no ops at all
            "itlb_miss": "L1_ITLB_MISS_L2_ITLB_MISS:COALESCED4K:IF1G:IF2M:IF4K",
        },
        [
            Metric("frontend_bound", lambda c: ratio(c["fe"], _slots(c)), "%"),
            Metric(
                "frontend_latency",
                lambda c: ratio(c["fe_lat"], c["cycles"]),
                "%",
                desc="i-cache/iTLB misses, branch redirects",
                hint=_hint(
                    lambda v: v > 0.1,
                    "frontend latency: large/branchy code, try PGO, -flto, "
                    "fewer indirect calls, huge pages for text",
                ),
            ),
            Metric(
                "frontend_bandwidth",
                lambda c: ratio(c["fe"] - ZEN5_SLOTS * c["fe_lat"], _slots(c)),
                "%",
                desc="decode/op cache throughput",
            ),
            Metric("itlb_mpki", _per_kilo("itlb_miss")),
        ],
    ),
    EventSet(
        "icache",
        "instruction and op cache",
        {
            "instructions": "RETIRED_INSTRUCTIONS",
            "ic_miss": "IC_TAG_HIT_MISS:IC_MISS",
            "ic_all": "IC_TAG_HIT_MISS:ALL_IC_ACCESS",
            "oc_miss": "OP_CACHE_HIT_MISS:OC_MISS",
            "oc_all": "OP_CACHE_HIT_MISS:ALL_OC_ACCESS",
        },
        [
            Metric("icache_miss_rate", lambda c: ratio(c["ic_miss"], c["ic_all"]), "%"),
            Metric("icache_mpki", _per_kilo("ic_miss")),
            Metric(
                "opcache_miss_rate",
                lambda c: ratio(c["oc_miss"], c["oc_all"]),
                "%",
                hint=_hint(
                    lambda v: v > 0.2,
                    "op cache misses: hot code footprint too large, check inlining "
                    "/ unrolling",
                ),
            ),
        ],
    ),
    EventSet(
        "backend",
        "backend stalls, memory vs core",
        {
            "cycles": "CYCLES_NOT_IN_HALT",
            "be": "DISPATCH_STALLS_1:BE_STALLS",
            "not_complete": "CYCLES_NO_RETIRE:NOT_COMPLETE_LOAD_AND_ALU",
            # libpfm lacks this umask combination
            "load_not_complete": "raw:cpu/event=0xd6,umask=0xa2/",
            "lq_stall": "DISPATCH_RESOURCE_STALL_CYCLES_1:LOAD_QUEUE_RSRC_STALL",
            "fp_sched_stall": "DISPATCH_RESOURCE_STALL_CYCLES_1:FP_SCHEDULER_RSRC_STALL",
        },
        [
            Metric("backend_bound", lambda c: ratio(c["be"], _slots(c)), "%"),
            Metric(
                "backend_memory",
                lambda c: ratio(c["be"], _slots(c))
                * ratio(c["load_not_complete"], c["not_complete"]),
                "%",
                hint=_hint(
                    lambda v: v > 0.2,
                    "memory bound: see -s cache,tlb; improve locality, blocking, "
                    "SoA layouts, prefetching",
                ),
            ),
            Metric(
                "backend_core",
                lambda c: ratio(c["be"], _slots(c))
                * (1 - ratio(c["load_not_complete"], c["not_complete"])),
                "%",
                desc="execution units / dependency chains / load-store throughput",
                hint=_backend_core_hint,
            ),
            Metric(
                "load_queue_stall",
                lambda c: ratio(c["lq_stall"], c["cycles"]),
                "%",
                desc="cycles dispatch stalled on a full load queue",
            ),
            Metric(
                "fp_scheduler_stall",
                lambda c: ratio(c["fp_sched_stall"], c["cycles"]),
                "%",
            ),
        ],
    ),
    EventSet(
        "cache",
        "L1D demand misses and where they were filled from",
        {
            "instructions": "RETIRED_INSTRUCTIONS",
            "loads": "LS_DISPATCH:LD_DISPATCH",
            "from_l2": f"{DDCF}:LCL_L2",
            "from_l3": f"{DDCF}:LOCAL_CCX",
            "from_dram": f"{DDCF}:DRAM_IO_NEAR:DRAM_IO_FAR:ALT_MEM_NEAR_FAR",
            "ram_fetches": RAM_FETCHES,
        },
        [
            Metric(
                "l1d_miss_rate",
                lambda c: ratio(
                    _sum("from_l2", "from_l3", "from_dram")(c),
                    c["loads"],
                ),
                "%",
                desc="demand fills / dispatched loads",
            ),
            Metric(
                "l1d_mpki",
                lambda c: ratio(
                    _sum("from_l2", "from_l3", "from_dram")(c),
                    c["instructions"] / 1000,
                ),
            ),
            Metric("l2_mpki", _per_kilo("from_l3"), desc="demand only"),
            Metric(
                "ram_prefetched",
                lambda c: ratio(c["ram_fetches"] - c["from_dram"], c["ram_fetches"]),
                "%",
                desc="RAM fetches brought in by prefetch rather than demand loads",
            ),
            Metric(
                "dram_mpki",
                _per_kilo("ram_fetches"),
                desc="lines fetched from RAM per kilo instruction, incl. prefetch",
                hint=_hint(
                    lambda v: v > 1,
                    "frequent DRAM fills: working set exceeds L3 or poor reuse, "
                    "consider tiling / streaming access order",
                ),
            ),
        ],
    ),
    EventSet(
        "tlb",
        "data TLB and misaligned access",
        {
            "instructions": "RETIRED_INSTRUCTIONS",
            "dtlb_l2_hit": "L1_DTLB_MISS:TLB_RELOAD_1G_L2_HIT:TLB_RELOAD_2M_L2_HIT"
            ":TLB_RELOAD_COALESCED_PAGE_HIT:TLB_RELOAD_4K_L2_HIT",
            "dtlb_l2_miss": "L1_DTLB_MISS:TLB_RELOAD_1G_L2_MISS:TLB_RELOAD_2M_L2_MISS"
            ":TLB_RELOAD_COALESCED_PAGE_MISS:TLB_RELOAD_4K_L2_MISS",
            "dtlb_4k_miss": "L1_DTLB_MISS:TLB_RELOAD_4K_L2_MISS",
            "split_line": "MISALIGNED_LOADS:MA64",
            "split_page": "MISALIGNED_LOADS:MA4K",
        },
        [
            Metric("l1_dtlb_mpki", _per_kilo("dtlb_l2_hit")),
            Metric(
                "l2_dtlb_mpki",
                _per_kilo("dtlb_l2_miss"),
                desc="page walks",
                hint=_hint(
                    lambda v: v > 0.5,
                    "page walks: try transparent huge pages "
                    "(madvise(MADV_HUGEPAGE) / THP always)",
                ),
            ),
            Metric(
                "page_walk_4k_fraction",
                lambda c: ratio(c["dtlb_4k_miss"], c["dtlb_l2_miss"]),
                "%",
            ),
            Metric(
                "cacheline_split_loads_pki",
                _per_kilo("split_line"),
                hint=_hint(lambda v: v > 1, "cache line split loads: align data"),
            ),
            Metric("page_split_loads_pki", _per_kilo("split_page")),
        ],
    ),
    EventSet(
        "vector",
        "FP/SIMD uops by vector width",
        {
            "x87": f"{BY_WIDTH}:X87_UOPS_RETIRED",
            "scalar": f"{BY_WIDTH}:SCALAR_UOPS_RETIRED",
            "p128": f"{BY_WIDTH}:PACK128_UOPS_RETIRED",
            "p256": f"{BY_WIDTH}:PACK256_UOPS_RETIRED",
            "p512": f"{BY_WIDTH}:PACK512_UOPS_RETIRED",
            "sse_avx_faults": "FP_DISPATCH_FAULTS:XMM_FILL_FAULT:YMM_FILL_FAULT:YMM_SPILL_FAULT",
        },
        [
            # BY_WIDTH counts every FP/SIMD pipe uop, including loads, stores and
            # register moves: a scalar movsd load counts as 128 bit. This is data
            # width, use flops_dp/flops_sp to judge if arithmetic is vectorized
            Metric(
                "packed_uops",
                lambda c: ratio(_sum("p128", "p256", "p512")(c), _fp_uops(c)),
                "%",
                desc="packed uops / all FP uops, includes loads/stores/moves",
            ),
            Metric(
                "scalar",
                lambda c: ratio(c["scalar"], _fp_uops(c)),
                "%",
                hint=_hint(
                    lambda v: v > 0.3,
                    "many scalar FP uops: run -s flops_dp,flops_sp to see if "
                    "the arithmetic is vectorized",
                ),
            ),
            Metric("width_128", lambda c: ratio(c["p128"], _fp_uops(c)), "%"),
            Metric(
                "width_256",
                lambda c: ratio(c["p256"], _fp_uops(c)),
                "%",
                hint=lambda v, c, ctx: (
                    "only 128-bit vectors: compiled for baseline x86-64? "
                    "try -march=native / -mavx2"
                    if c.get("p128") and not c.get("p256") and not c.get("p512")
                    else None
                ),
            ),
            Metric(
                "width_512",
                lambda c: ratio(c["p512"], _fp_uops(c)),
                "%",
                hint=_avx512_hint,
            ),
            Metric(
                "x87",
                lambda c: ratio(c["x87"], _fp_uops(c)),
                "%",
                hint=_hint(
                    lambda v: v > 0.01,
                    "x87 uops: long double math or -mfpmath=387, these do not vectorize",
                ),
            ),
            Metric(
                "sse_avx_faults",
                lambda c: c["sse_avx_faults"],
                hint=_hint(
                    lambda v: v > 0,
                    "SSE/AVX dispatch faults: mixing legacy SSE and AVX, "
                    "missing vzeroupper or non-VEX library code",
                ),
            ),
        ],
    ),
    EventSet(
        "flops",
        "SSE/AVX floating point operations",
        {
            "cycles": "CYCLES_NOT_IN_HALT",
            "flops": f"{FLOPS}:ANY",
            "mac_flops": f"{FLOPS}:MAC_FLOPS",  # each FMA counts as 2
        },
        [
            Metric("flops_per_cycle", lambda c: ratio(c["flops"], c["cycles"])),
            Metric(
                "fma_fraction",
                lambda c: ratio(c["mac_flops"], c["flops"]),
                "%",
                desc="flops from fused multiply add",
                hint=lambda v, c, ctx: (
                    "few FMAs among many flops: allow contraction "
                    "(-ffp-contract=fast, -march=native)"
                    if v is not None and v < 0.25 and c["flops"] > 1e6
                    else None
                ),
            ),
        ],
    ),
    EventSet(
        "flops_dp",
        "double precision flops, scalar vs packed",
        {
            "cycles": "CYCLES_NOT_IN_HALT",
            "scalar_dp": f"{FLOPS}:ANY:FLOP_TYPE_SCALAR_DOUBLE",
            "packed_dp": f"{FLOPS}:ANY:FLOP_TYPE_PACKED_DOUBLE",
        },
        [
            Metric(
                "dp_flops_per_cycle",
                lambda c: ratio(_sum("scalar_dp", "packed_dp")(c), c["cycles"]),
            ),
            Metric(
                "dp_packed",
                lambda c: ratio(c["packed_dp"], _sum("scalar_dp", "packed_dp")(c)),
                "%",
                hint=_hint(
                    lambda v: v < 0.5,
                    "most double precision flops are scalar, not vectorized: check "
                    + VECTORIZE_ADVICE,
                ),
            ),
        ],
    ),
    EventSet(
        "flops_sp",
        "single precision flops, scalar vs packed",
        {
            "cycles": "CYCLES_NOT_IN_HALT",
            "scalar_sp": f"{FLOPS}:ANY:FLOP_TYPE_SCALAR_SINGLE",
            "packed_sp": f"{FLOPS}:ANY:FLOP_TYPE_PACKED_SINGLE",
        },
        [
            Metric(
                "sp_flops_per_cycle",
                lambda c: ratio(_sum("scalar_sp", "packed_sp")(c), c["cycles"]),
            ),
            Metric(
                "sp_packed",
                lambda c: ratio(c["packed_sp"], _sum("scalar_sp", "packed_sp")(c)),
                "%",
                hint=_hint(
                    lambda v: v < 0.5,
                    "most single precision flops are scalar, not vectorized: check "
                    + VECTORIZE_ADVICE,
                ),
            ),
        ],
    ),
]


# --------------------------------------------------------------------------- #
# any CPU: perf generic hardware events

# CPUs without sets above: perf's generic hardware events, which the kernel maps
# to each CPU's own, so they count on any x86 or arm64 with a PMU. What they
# count differs per CPU (see GENERIC_ACCURACY), sets only, no topdown or flops
PERF_HW_IDS = {  # perf_event_attr.config for PERF_TYPE_HARDWARE
    "cycles": 0,
    "instructions": 1,
    "cache-references": 2,
    "cache-misses": 3,
    "branches": 4,
    "branch-misses": 5,
}

generic = [
    EventSet(
        "light",
        "perf generic events: cycles, instructions, branches",
        {
            "cycles": "perf:cycles",
            "instructions": "perf:instructions",
            "branches": "perf:branches",
            "br_misp": "perf:branch-misses",
        },
        [
            Metric(
                "ipc",
                lambda c: ratio(c["instructions"], c["cycles"]),
                hint=_hint(lambda v: v < 1, "low IPC"),
            ),
            Metric(
                "branch_mispredict_rate",
                lambda c: ratio(c["br_misp"], c["branches"]),
                "%",
            ),
        ],
    ),
    EventSet(
        "ipc",
        "perf generic events",
        {
            "cycles": "perf:cycles",
            "instructions": "perf:instructions",
            "branches": "perf:branches",
            "br_misp": "perf:branch-misses",
            "cache_refs": "perf:cache-references",
            "cache_misses": "perf:cache-misses",
        },
        [
            Metric(
                "ipc",
                lambda c: ratio(c["instructions"], c["cycles"]),
                hint=_hint(lambda v: v < 1, "low IPC"),
            ),
            Metric(
                "branch_mispredict_rate",
                lambda c: ratio(c["br_misp"], c["branches"]),
                "%",
            ),
            Metric(
                "cache_miss_rate",
                lambda c: ratio(c["cache_misses"], c["cache_refs"]),
                "%",
            ),
        ],
    ),
]


# --------------------------------------------------------------------------- #
# arm64, architected PMUv3 events. Formulas follow linux
# tools/perf/pmu-events/arch/arm64/arm/neoverse-n2-v2/metrics.json, "@slots" is
# the core PMU's caps/slots, see host.context()


def _arm_slots(c):
    return c["cycles"] * c["@slots"] if c.get("@slots") else None


def _arm_stall(c, key):
    """N2 r0p0/r0p3 STALL_SLOT* count one extra slot per cycle"""
    return c[key] - c["cycles"] if c.get("@stall_slot_counts_cycles") else c[key]


def _arm_not_stalled(c):
    return 1 - ratio(_arm_stall(c, "stall"), _arm_slots(c))


arm_light = EventSet(
    "light",
    "cheap per scope counters: cycles, instructions, RAM fetches",
    {
        "cycles": "CPU_CYCLES",
        "instructions": "INST_RETIRED",
        # last level cache read misses: RAM. Only counts when the core is
        # configured with an external LLC (CPUECTLR.EXTLLC), check diagnose
        "fetches": "LL_CACHE_MISS_RD",
    },
    [
        Metric("ipc", lambda c: ratio(c["instructions"], c["cycles"])),
        Metric(
            "instructions_per_fetch",
            lambda c: ratio(c["instructions"], c["fetches"]),
            desc="instructions per line read from RAM",
            none_means=_no_fetches,
            hint=_hint(
                lambda v: v < 100,
                "under 100 instructions per RAM fetch: memory bound, improve reuse",
            ),
        ),
        Metric(
            "fetch_bytes_per_cycle",
            lambda c: ratio(c["fetches"] * 64, c["cycles"]),
            desc="assumes 64B lines",
        ),
    ],
)

arm_ipc = EventSet(
    "ipc",
    "instructions per cycle and branch prediction",
    {
        "cycles": "CPU_CYCLES",
        "instructions": "INST_RETIRED",
        "spec_instructions": "INST_SPEC",
        "branches": "BR_RETIRED",
        "br_misp": "BR_MIS_PRED_RETIRED",
    },
    [
        Metric(
            "ipc",
            lambda c: ratio(c["instructions"], c["cycles"]),
            hint=_hint(lambda v: v < 1, "low IPC: see -s topdown"),
        ),
        Metric(
            "branch_mispredict_rate",
            lambda c: ratio(c["br_misp"], c["branches"]),
            "%",
            hint=_hint(lambda v: v > 0.02, "branch mispredicts > 2%"),
        ),
        Metric("branch_mpki", _per_kilo("br_misp")),
        Metric(
            "speculation_wasted",
            lambda c: 1 - ratio(c["instructions"], c["spec_instructions"]),
            "%",
            desc="speculatively executed instructions that did not retire",
        ),
    ],
)

arm_topdown = EventSet(
    "topdown",
    "level 1 pipeline utilization, fraction of issue slots",
    {
        "cycles": "CPU_CYCLES",
        "stall": "STALL_SLOT",
        "stall_fe": "STALL_SLOT_FRONTEND",
        "stall_be": "STALL_SLOT_BACKEND",
        "op_spec": "OP_SPEC",
        "op_retired": "OP_RETIRED",
        "br_misp_spec": "BR_MIS_PRED",
    },
    [
        Metric(
            "retiring",
            lambda c: ratio(c["op_retired"], c["op_spec"]) * _arm_not_stalled(c),
            "%",
            hint=_hint(
                lambda v: v > 0.5,
                "mostly retiring: do less work, vectorize (-s vector,flops)",
            ),
        ),
        Metric(
            "frontend_bound",
            lambda c: ratio(_arm_stall(c, "stall_fe"), _arm_slots(c))
            - ratio(c["br_misp_spec"], c["cycles"]),
            "%",
            hint=_hint(lambda v: v > 0.2, "frontend bound: i-cache/iTLB, code size"),
        ),
        Metric(
            "backend_bound",
            lambda c: ratio(c["stall_be"], _arm_slots(c))
            - ratio(c["br_misp_spec"] * 3, c["cycles"]),
            "%",
            hint=_hint(lambda v: v > 0.3, "backend bound: see -s cache"),
        ),
        Metric(
            "bad_speculation",
            lambda c: (1 - ratio(c["op_retired"], c["op_spec"])) * _arm_not_stalled(c)
            + ratio(c["br_misp_spec"] * 4, c["cycles"]),
            "%",
            hint=_hint(lambda v: v > 0.1, "bad speculation: see -s ipc"),
        ),
        Metric("slots", lambda c: c.get("@slots"), desc="from caps/slots"),
    ],
)


def _arm_vector(with_sve):
    simd = ("neon", "sve") if with_sve else ("neon",)
    events = {
        "spec_instructions": "INST_SPEC",
        "neon": "ASE_SPEC",
        "scalar_fp": "VFP_SPEC",
    }
    if with_sve:
        events["sve"] = "SVE_INST_SPEC"
    metrics = [
        Metric(
            "vectorized",
            lambda c: ratio(_sum(*simd)(c), _sum(*simd, "scalar_fp")(c)),
            "%",
            desc="SIMD / (SIMD + scalar FP) instructions, SIMD includes integer",
            hint=_hint(
                lambda v: v < 0.5,
                "mostly scalar floating point: check " + VECTORIZE_ADVICE,
            ),
        ),
        Metric("neon", lambda c: ratio(c["neon"], c["spec_instructions"]), "%"),
        Metric(
            "scalar_fp", lambda c: ratio(c["scalar_fp"], c["spec_instructions"]), "%"
        ),
    ]
    if with_sve:
        metrics.append(
            Metric(
                "sve",
                lambda c: ratio(c["sve"], c["spec_instructions"]),
                "%",
                hint=lambda v, c, ctx: (
                    "SVE available but unused: try -mcpu=native / -march=armv8-a+sve"
                    if "sve" in ctx.get("cpu_flags", ())
                    and not c.get("sve")
                    and c.get("neon")
                    else None
                ),
            )
        )
    return EventSet(
        "vector", "SIMD (NEON/SVE) vs scalar FP instructions", events, metrics
    )


def _arm_sve_flops(c):
    """FP_SCALE_OPS_SPEC counts per 128 bits of SVE vector length. Assumes the
    measured program runs at @sve_bits, see the sve_bits metric"""
    bits = c.get("@sve_bits") or 128
    return c["scale_ops"] * bits / 128 + c["fixed_ops"]


arm_flops = EventSet(
    "flops",
    "floating point operations, SVE scaled by vector length",
    {
        "cycles": "CPU_CYCLES",
        "scale_ops": "FP_SCALE_OPS_SPEC",
        "fixed_ops": "FP_FIXED_OPS_SPEC",
    },
    [
        Metric("flops_per_cycle", lambda c: ratio(_arm_sve_flops(c), c["cycles"])),
        Metric(
            "scalable_fraction",
            lambda c: ratio(c["scale_ops"], _sum("scale_ops", "fixed_ops")(c)),
            "%",
            desc="SVE (vector length scaled) share of FP ops, speculative",
        ),
        # the system default length (/proc/sys/abi/sve_default_vector_length):
        # wrong for programs that change theirs with prctl(PR_SVE_SET_VL), set
        # sve_bits in PHLOP_PERF_CTX / the timer file's #ctx line for those
        Metric(
            "sve_bits",
            lambda c: c.get("@sve_bits"),
            desc="SVE length assumed, the system default unless overridden",
        ),
    ],
)

arm_cache = EventSet(
    "cache",
    "cache refills and last level misses",
    {
        "instructions": "INST_RETIRED",
        "mem_access": "MEM_ACCESS",
        "l1d_refill": "L1D_CACHE_REFILL",
        "l2d_refill": "L2D_CACHE_REFILL",
        "ll_read": "LL_CACHE_RD",
        "ll_miss_read": "LL_CACHE_MISS_RD",
    },
    [
        Metric("l1d_miss_rate", lambda c: ratio(c["l1d_refill"], c["mem_access"]), "%"),
        Metric("l1d_mpki", _per_kilo("l1d_refill")),
        Metric("l2d_mpki", _per_kilo("l2d_refill")),
        Metric(
            "ll_read_miss_rate", lambda c: ratio(c["ll_miss_read"], c["ll_read"]), "%"
        ),
        Metric(
            "ram_mpki",
            _per_kilo("ll_miss_read"),
            hint=_hint(
                lambda v: v > 1,
                "frequent RAM reads: working set exceeds caches, improve reuse",
            ),
        ),
    ],
)

arm_v9ish = [arm_light, arm_ipc, arm_topdown, _arm_vector(True), arm_flops, arm_cache]
arm_basic = [arm_light, arm_ipc, _arm_vector(False), arm_cache]


SETS = {
    "amd64_fam1ah_zen5": zen5,
    **{pmu: arm_v9ish for pmu in ("arm_v1", "arm_n2", "arm_v2", "arm_n3", "arm_v3")},
    **{pmu: arm_basic for pmu in ("arm_n1", "arm_ac76")},
}
DEFAULT_SETS = ["ipc", "topdown", "vector", "flops", "cache"]


def sets_for(pmus):
    """(pmu, {name: EventSet}) for the first detected PMU with known sets"""
    for pmu in pmus:
        if pmu in SETS:
            return pmu, {s.name: s for s in SETS[pmu]}
    return "generic", {s.name: s for s in generic}


# --------------------------------------------------------------------------- #
# Warnings when counts are less trustworthy than with a host's own sets: what
# the problem is, why the numbers may be off, and how to resolve it

GENERIC_ACCURACY = (
    "potentially inaccurate: the kernel maps generic events to each CPU's own, "
    "so what they count differs between CPUs, eg cache-references/misses may be "
    "last level cache or another level, branches may include or exclude some "
    "branch types, and VMs may filter or approximate them. Compare runs on the "
    "same CPU model only, treat cache metrics as indicative"
)
FORCED_ACCURACY = (
    "potentially wrong: this CPU's counters are programmed with another model's "
    "event codes, events may count something else or nothing, metrics follow"
)


def pmu_warnings(pmu, detected, forced=None, host=None):
    """warnings for the sets sets_for picked. pmu: its pick, detected: libpfm's
    PMUs (the forced one if LIBPFM_FORCE_PMU), forced: LIBPFM_FORCE_PMU, host:
    libpfm's PMUs for this CPU, ignoring the force"""
    known = ", ".join(SETS)
    warnings = []
    if pmu == "generic":
        if detected:
            what = f"no phlop event sets for PMU {detected[0]}"
            fix = (
                f"add event sets for {detected[0]} to SETS in phlop/app/pfm/sets.py "
                f"(sets exist for {known}), check their event names offline with "
                f"LIBPFM_FORCE_PMU={detected[0]} python3 -m phlop.app.pfm.stat --list"
            )
        elif forced:
            what = f"LIBPFM_FORCE_PMU={forced} is not a PMU model this libpfm knows"
            fix = (
                "unset it, or use a model from tpp/pfm/examples/showevtinfo. A "
                "libpfm build knows its own architecture's models only, eg no arm "
                "models on x86"
            )
        else:
            what = "libpfm detected no PMU model for this CPU"
            fix = (
                "build libpfm: ./sh/setup_pfm.sh (or set PHLOP_PFM_DIR). If built, "
                "libpfm is too old for this CPU (update it) or this is a VM without "
                "a PMU (python3 -m phlop.app.pfm.diagnose)"
            )
        warnings.append(
            f"{what}, using perf generic events (cycles, instructions, branches, "
            f"cache)\n  {GENERIC_ACCURACY}\n  resolve: {fix}"
        )
    elif forced and forced not in (host or []):
        this = (
            f"this CPU is {host[0]}" if host else "libpfm knows no model for this CPU"
        )
        warnings.append(
            f"LIBPFM_FORCE_PMU={forced}: events are encoded for {forced}, {this}\n"
            f"  {FORCED_ACCURACY}\n  resolve: unset LIBPFM_FORCE_PMU to count with "
            "this CPU's own events. Forcing is for checking a model's event names "
            "and encodings: --list, --env, -p"
        )
    return warnings


def scope_file_warnings(pmus, forced):
    """warnings for a scope timer file: pmus, its "#perf" level labels' PMUs,
    forced: its "#ctx" says the events were encoded for a forced PMU"""
    warnings = []
    if "generic" in pmus:
        warnings.append(
            "counted with perf generic events, there were no phlop event sets for "
            f"the measuring CPU: {GENERIC_ACCURACY}. Resolve: add sets for that "
            "CPU in phlop/app/pfm/sets.py"
        )
    if forced:
        warnings.append(
            "counted with events encoded for a forced libpfm PMU (LIBPFM_FORCE_PMU) "
            f"other than the measuring CPU: {FORCED_ACCURACY}. Resolve: count again "
            "with PHLOP_PERF_* generated without LIBPFM_FORCE_PMU"
        )
    return warnings
