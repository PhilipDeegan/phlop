#
#
#


from phlop.proc import run
from phlop.string import decode_bytes

if __name__ == "__main__":
    from phlop.app.pfm.check_events import get_evt_perf_code
    from phlop.app.pfm.showevtinfo import get_evt_info

    code = ""
    key0, key1 = "[MULT_FLOPS]", "[ADD_SUB_FLOPS]"
    for info in get_evt_info():
        if key0 in info.umask:
            for key, umask in info.umask.items():
                code += f"{info.name}:{umask.code} "
            break
        # if key1 in info.umask:
        #     code += f"{info.name}:{info.umask[key1].code} "

    code = code.strip()
    assert code != ""

    events = " ".join([f"-e {get_evt_perf_code(ev)}" for ev in code.split(" ")])
    print(decode_bytes(run(f"perf stat {events} sleep 2").stderr))
