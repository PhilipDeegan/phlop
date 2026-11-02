# tests/_phlop/app/pfm/test_pfm.py
#
# needs libpfm, see ./sh/setup_pfm.sh

from phlop.app import perf
from phlop.app.pfm.check_events import encode, perf_event_string
from phlop.app.pfm.showevtinfo import get_evt_info
from phlop.proc import run
from phlop.string import decode_bytes

if __name__ == "__main__":
    # intel: FP_ARITH_INST_RETIRED style umasks, amd zen4/5: RETIRED_SSE_AVX_FLOPS
    key0 = "[MULT_FLOPS]"
    codes = []
    for info in get_evt_info():
        if key0 in info.umask:
            codes = [f"{info.name}:{umask.id[1:-1]}" for umask in info.umask.values()]
            break

    assert codes, f"no PMU event with {key0} umask"

    encoded = encode(codes)
    events = " ".join(
        f"-e {perf_event_string(f'ev{i}', cfg, 'u')}"
        for i, cfg in enumerate(encoded.values())
    )
    print(events)

    can_count, warnings = perf.access_warnings()
    for w in warnings:
        print("WARNING:", w)
    if can_count:
        print(decode_bytes(run(f"perf stat {events} sleep 2").stderr))
