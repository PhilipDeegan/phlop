# tests/_phlop/app/nvidia/test_cuobjdump.py
#
# the block size limit with known answers, the cuobjdump parser on its output,
# then a real binary if nvcc is there (no GPU needed)

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from phlop.app.nvidia import cuobjdump as cu

OUTPUT = """
Fatbin elf code:
================
arch = sm_89
code version = [1,8]

Resource usage:
 Common:
  GLOBAL:0
 Function _Z5heavyILi64EEvPf:
  REG:148 STACK:16 SHARED:0 LOCAL:24 CONSTANT[0]:360 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z5lightPf:
  REG:8 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:360 TEXTURE:0 SURFACE:0 SAMPLER:0

Fatbin elf code:
================
arch = sm_120

Resource usage:
 Common:
  GLOBAL:0
 Function _Z5lightPf:
  REG:10 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:904 TEXTURE:0 SURFACE:0 SAMPLER:0
"""

SOURCE = """
__global__ void light(float* y) { y[threadIdx.x] = 1; }
int main() { light<<<1, 1>>>(nullptr); }
"""


class LimitTest(unittest.TestCase):
    def test_max_threads(self):
        # PHARE's debug build: 148-172 registers can't launch 512, release 86 fits 736
        expect = {
            0: 1024,
            32: 1024,
            64: 1024,
            72: 896,
            86: 736,
            148: 416,
            172: 352,
            255: 256,
        }
        self.assertEqual({r: cu.max_threads(r) for r in expect}, expect)
        self.assertEqual(cu.max_threads(64, regs_per_block=32768), 512)

    def test_pow2_floor(self):
        self.assertEqual(
            [cu.pow2_floor(n) for n in (0, 1, 416, 512, 736)], [0, 1, 256, 512, 512]
        )


class ParseTest(unittest.TestCase):
    def test_parse(self):
        ks = cu.parse(OUTPUT)
        self.assertEqual(
            [(k.mangled, k.arch, k.reg) for k in ks],
            [("_Z5heavyILi64EEvPf", "sm_89", 148), ("_Z5lightPf", "sm_89", 8),
             ("_Z5lightPf", "sm_120", 10)],
        )  # fmt: skip
        self.assertEqual((ks[0].stack, ks[0].local, ks[1].shared), (16, 24, 1024))

    def test_shorten(self):
        name = "void ns::k<A<B<int>>, 3>(float*)"
        self.assertEqual(cu.shorten(name, 0), "void ns::k<…>(float*)")
        self.assertEqual(cu.shorten(name, 1), "void ns::k<A<…>, 3>(float*)")

    def test_report(self):
        with mock.patch.object(cu, "kernels", return_value=cu.parse(OUTPUT)):
            args = cu.cli_args_parser().parse_args(["-t", "512", "dbg=a"])
            text = cu.report(args)
        self.assertIn("== dbg (a): 3 kernels, 1 can't launch 512 threads", text)
        self.assertIn(
            " 148     16      0     24    416  sm_89    _Z5heavyILi64EEvPf", text
        )
        self.assertIn("       3    148    416   256       1  dbg", text)  # summary


@unittest.skipUnless(shutil.which("nvcc") and shutil.which("cuobjdump"), "needs nvcc")
class RealBinaryTest(unittest.TestCase):
    def test_binary(self):
        tmp = tempfile.mkdtemp()
        try:
            src, exe = os.path.join(tmp, "k.cu"), os.path.join(tmp, "k")
            with open(src, "w") as f:
                f.write(SOURCE)
            subprocess.run(["nvcc", "-o", exe, src], check=True)
            ks = cu.kernels(exe)
            demangled = shutil.which("llvm-cxxfilt") or shutil.which("c++filt")
            light = "light(float*)" if demangled else "_Z5lightPf"  # else mangled
            self.assertIn(light, {k.name for k in ks})
            self.assertTrue(all(k.reg > 0 for k in ks))
            res = subprocess.run(
                [sys.executable, "-m", "phlop.app.nvidia.cuobjdump", sys.executable],
                capture_output=True, text=True, check=False,
            )  # fmt: skip
            self.assertEqual(res.returncode, 1)
            self.assertIn("does not contain device code", res.stderr)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
