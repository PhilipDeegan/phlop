# tests/_phlop/app/valgrind/test_valgrind.py
#
# parsers and reports on synthetic files, then real runs if valgrind and a
# c++ compiler are available

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from phlop.app import valgrind as vg
from phlop.app.valgrind import massif, memcheck, report

MASSIF = """desc: --time-unit=ms --massif-out-file=m.%p
cmd: ./alloc
time_unit: ms
#-----------
snapshot=0
#-----------
time=0
mem_heap_B=0
mem_heap_extra_B=0
mem_stacks_B=0
heap_tree=empty
#-----------
snapshot=1
#-----------
time=150
mem_heap_B=10559488
mem_heap_extra_B=4048
mem_stacks_B=0
heap_tree=empty
#-----------
snapshot=2
#-----------
time=160
mem_heap_B=42016768
mem_heap_extra_B=8088
mem_stacks_B=0
heap_tree=peak
n3: 42016768 (heap allocation functions) malloc/new/new[], --alloc-fns, etc.
 n1: 31457280 0x1091BF: allocate (new_allocator.h:151)
  n1: 31457280 0x1091BF: _M_allocate (stl_vector.h:380)
   n1: 31457280 0x1091BF: vector (stl_vector.h:571)
    n2: 31457280 0x1091BF: big(unsigned long) (alloc.cpp:5)
     n0: 20971520 0x10926D: main (alloc.cpp:11)
     n0: 10485760 0x109260: main (alloc.cpp:9)
 n1: 10485760 0x4A2C: H5D__chunk_collective_fill (in /usr/lib/libhdf5.so.310)
  n0: 10485760 0x109300: write (h5.cpp:20)
 n0: 73728 in 2 places, below massif's threshold (1.00%)
#-----------
snapshot=3
#-----------
time=170
mem_heap_B=0
mem_heap_extra_B=0
mem_stacks_B=0
heap_tree=empty
"""

MEMCHECK = """<?xml version="1.0"?>
<valgrindoutput>
<protocolversion>4</protocolversion>
<protocoltool>memcheck</protocoltool>
<pid>1234</pid>
<args><argv><exe>./bad</exe><arg>x</arg></argv></args>
<error>
  <unique>0x1</unique>
  <kind>InvalidRead</kind>
  <what>Invalid read of size 4</what>
  <stack>
    <frame><fn>sum(int const*, int)</fn><file>bad.cpp</file><line>4</line></frame>
    <frame><fn>main</fn><file>bad.cpp</file><line>9</line></frame>
  </stack>
  <auxwhat>Address 0x4e0c0a8 is 0 bytes after a block of size 40 alloc'd</auxwhat>
  <stack>
    <frame><fn>operator new(unsigned long)</fn><file>vg_replace_malloc.c</file><line>487</line></frame>
    <frame><fn>std::__new_allocator&lt;int&gt;::allocate(unsigned long, void const*)</fn><file>new_allocator.h</file><line>151</line></frame>
    <frame><fn>main</fn><file>bad.cpp</file><line>6</line></frame>
  </stack>
</error>
<error>
  <unique>0x2</unique>
  <kind>Leak_DefinitelyLost</kind>
  <xwhat><text>400 bytes in 1 blocks are definitely lost</text>
    <leakedbytes>400</leakedbytes><leakedblocks>1</leakedblocks></xwhat>
  <stack>
    <frame><fn>operator new[](unsigned long)</fn><file>vg_replace_malloc.c</file><line>729</line></frame>
    <frame><fn>main</fn><file>bad.cpp</file><line>10</line></frame>
  </stack>
</error>
<errorcounts>
  <pair><count>3</count><unique>0x1</unique></pair>
</errorcounts>
</valgrindoutput>
"""


def _write(directory, name, text):
    path = Path(directory) / name
    path.write_text(text)
    return str(path)


class MassifTest(unittest.TestCase):
    def setUp(self):
        self.m = massif.parse_lines(MASSIF.splitlines(keepends=True), "m.p1.out")

    def test_parse(self):
        m = self.m
        self.assertEqual((m.cmd, m.time_unit, m.threshold), ("./alloc", "ms", 1.0))
        self.assertEqual(len(m.snapshots), 4)
        self.assertEqual(m.peak.time, 160)
        self.assertEqual(m.peak.total, 42016768 + 8088)
        self.assertEqual(m.peak_tree.kind, "peak")

    def test_sites_collapse_runtime_frames(self):
        sites = massif.sites(self.m.peak_tree.tree)
        # vector's allocate/_M_allocate/vector skipped to big(), hdf5 is kept
        self.assertEqual(
            [s.frame.fn if s.frame else None for s in sites],
            ["big(unsigned long)", "H5D__chunk_collective_fill", None],
        )
        big = sites[0]
        self.assertEqual(big.bytes, 31457280)
        self.assertEqual([c.frame.line for c in big.callers], [11, 9])
        self.assertEqual((sites[-1].bytes, sites[-1].below), (73728, 2))

    def test_text(self):
        text = report.text([self.m])
        first = text.splitlines()[1]
        self.assertIn("peak 40.1MiB at 160ms of 170ms", first)
        self.assertIn("from big(unsigned long) (alloc.cpp:5)", first)
        self.assertIn("20.0MiB from main (alloc.cpp:11)", text)
        self.assertIn("2 places below massif's threshold", text)

    def test_processes_and_compare(self):
        small = MASSIF.replace("31457280", "1048576").replace("42016768", "11534336")
        other = massif.parse_lines(small.splitlines(keepends=True), "m.p2.out")
        text = report.text([other, self.m])
        self.assertIn("2 processes", text.splitlines()[0])
        self.assertIn("worst pid 1", text.splitlines()[0])

        no_hdf5 = MASSIF.replace("H5D__chunk_collective_fill", "fill")
        b = massif.parse_lines(no_hdf5.splitlines(keepends=True), "m.p3.out")
        cmp = report.text([self.m], compare=[b], names=("master", "pr"))
        self.assertIn("pr - master = +0B", cmp)
        self.assertIn("at master's worst peak, in no pr profile", cmp)
        self.assertIn("H5D__chunk_collective_fill", cmp)

    def test_markdown(self):
        md = report.markdown([self.m])
        self.assertTrue(md.startswith("**peak 40.1MiB"))
        self.assertIn("xychart-beta", md)
        self.assertIn("flowchart BT", md)
        self.assertIn("`big(unsigned long) (alloc.cpp:5)`", md)


class MemcheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = memcheck.parse(_write(self.tmp.name, "memcheck.r.p1234.xml", MEMCHECK))

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse(self):
        m = self.m
        self.assertEqual((m.pid, m.cmd), (1234, "./bad x"))
        self.assertEqual([e.count for e in m.problems], [3])
        self.assertEqual(m.leaked("Leak_DefinitelyLost"), 400)
        self.assertEqual(m.problems[0].site.fn, "sum(int const*, int)")
        self.assertEqual(m.leaks[0].site.fn, "main")  # not operator new

    def test_text(self):
        text = report.text([self.m])
        self.assertIn("3 errors, definitely lost 400B", text)
        self.assertIn("at sum(int const*, int) (bad.cpp:4) ← main (bad.cpp:9)", text)
        self.assertIn("alloc'd: main (bad.cpp:6)", text)  # runtime frames skipped

    def test_truncated(self):  # the process died: no closing tag
        cut = MEMCHECK.replace("</valgrindoutput>\n", "")
        m = memcheck.parse(_write(self.tmp.name, "memcheck.r.p2.xml", cut))
        self.assertEqual(len(m.errors), 2)

    def test_clean(self):
        clean = MEMCHECK.split("<error>")[0] + "</valgrindoutput>\n"
        m = memcheck.parse(_write(self.tmp.name, "memcheck.r.p3.xml", clean))
        self.assertEqual(report.memcheck_headline(m), "clean: no errors, no leaks")


class ResolveTest(unittest.TestCase):
    def test_run_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "massif.20260101-000000.p1.out", MASSIF)
            _write(tmp, "massif.20260101-000000.p1.log", "")
            newer = _write(tmp, "memcheck.20260101-000001.p2.xml", MEMCHECK)
            os.utime(newer, (2e9, 2e9))
            self.assertEqual(vg.resolve(None, tmp), [newer])  # latest run
            self.assertEqual(len(vg.resolve("20260101-000000", tmp)), 1)
            self.assertEqual(vg.resolve(tmp, tmp), [newer])
            with self.assertRaises(FileNotFoundError):
                vg.resolve("nope", tmp)


SOURCE = """#include <cstdlib>
#include <vector>
int sum(int const* p, int n){ int s = 0; for (int i = 0; i <= n; ++i) s += p[i]; return s; }
std::vector<char> big(std::size_t n){ return std::vector<char>(n, 1); }
int main(int argc, char** argv){
  if (argc > 1) { auto v = big(std::size_t{16} << 20); return v[0] == 1 ? 0 : 1; }
  std::vector<int> v(10, 1);
  int* lost = new int[100]; lost[0] = sum(v.data(), 10);
  return 0;
}
"""


def _cxx():
    return shutil.which(os.environ.get("CXX", "c++")) or shutil.which("g++")


@unittest.skipUnless(shutil.which("valgrind") and _cxx(), "needs valgrind and c++")
class RunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        src = _write(cls.tmp.name, "prog.cpp", SOURCE)
        cls.exe = str(Path(cls.tmp.name) / "prog")
        subprocess.run([_cxx(), "-g", "-O0", src, "-o", cls.exe], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_memcheck(self):
        outdir = Path(self.tmp.name) / "memcheck"
        rc, run_id = vg.run("memcheck", [self.exe], outdir=outdir, quiet=True)
        self.assertEqual(rc, vg.ERROR_EXITCODE)
        text = report.text(vg.results(run_id, outdir))
        self.assertIn("Invalid read of size 4", text)
        self.assertIn("definitely lost 400B", text)

    def test_massif(self):
        outdir = Path(self.tmp.name) / "massif"
        rc, run_id = vg.run("massif", [self.exe, "big"], outdir=outdir, quiet=True)
        self.assertEqual(rc, 0)
        (m,) = vg.results(run_id, outdir)
        self.assertGreaterEqual(m.peak.heap, 16 << 20)
        self.assertEqual(
            massif.sites(m.peak_tree.tree)[0].frame.fn, "big(unsigned long)"
        )


if __name__ == "__main__":
    unittest.main()
