#
#
#
#
#


import json
from dataclasses import dataclass, field

from phlop.proc import run
from phlop.string import decode_bytes


def version():
    pass


def make_config_str(path, cxx_flags=None, use_ninja=False, use_ccache=False, extra=""):
    cxx_flags = "" if cxx_flags is None else f'-DCMAKE_CXX_FLAGS="{cxx_flags}"'
    ccache = "" if use_ccache is False else "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    ninja = "" if not use_ninja else "-G Ninja"

    return f"cmake {path} {cxx_flags} {ninja} {ccache} {extra}"


def config(path, cxx_flags=None, use_ninja=False, use_ccache=False, extra=""):
    cmd = make_config_str(path, cxx_flags, use_ninja, extra)
    run(cmd, capture_output=False)


def build(use_ninja=False, threads=1):
    run("ninja" if use_ninja else f"make -j{threads}", capture_output=False)


@dataclass
class CTest_test:
    backtrace: int  # care?
    command: list  # list[str] # eventually
    # [
    #   "/opt/py/py/bin/python3",
    #   "-u",
    #   "test_particles_advance_2d.py"
    # ],
    name: str  # "py3_advance-2d-particles",
    properties: list = field(default_factory=lambda: [{}])  # list[dict] # eventually

    env: dict = field(default_factory=lambda: {})  # dict[str, str] # eventually
    working_dir: str = field(default_factory=lambda: None)

    def __post_init__(self):
        for p in self.properties:
            if p["name"] == "ENVIRONMENT":
                for item in p["value"]:
                    bits = item.split("=")
                    self.env.update({bits[0]: "=".join(bits[1:])})
            elif p["name"] == "WORKING_DIRECTORY":
                self.working_dir = p["value"]

    # [
    #   {
    #     "name" : "ENVIRONMENT",
    #     "value" :
    #     [
    #       "PYTHONPATH=/home/p/git/phare/master/build:/home/p/git/phare/master/pyphare",
    #       "ASAN_OPTIONS=detect_leaks=0"
    #     ]
    #   },
    #   {
    #     "name" : "WORKING_DIRECTORY",
    #     "value" : "/home/p/git/phare/master/build/tests/simulator/advance"
    #   }
    # ]


def list_tests(build_dir=None):
    cmd = "".join(
        [
            s
            for s in [
                "ctest ",
                f"--test-dir {build_dir} " if build_dir else "",
                "--show-only=json-v1",
            ]
            if s
        ]
    )
    return [
        CTest_test(**test)
        for test in json.loads(decode_bytes(run(cmd, capture_output=True).stdout))[
            "tests"
        ]
    ]


def test_cmd(test, verbose=False):
    cmd = f"ctest -R {test}"
    if verbose:
        cmd = f"{cmd} -V"
    return cmd


if __name__ == "__main__":
    list_tests("build")
