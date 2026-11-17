# phlop/app/valgrind/common.py
#
# shared by the valgrind tool parsers and reports

import os
import re
from dataclasses import dataclass


def format_bytes(b, sign=False):
    v = float(abs(b))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if v < 1024 or unit == "TiB":
            break
        v /= 1024
    s = "-" if b < 0 else ("+" if sign else "")
    return f"{s}{v:.0f}{unit}" if unit == "B" else f"{s}{v:.1f}{unit}"


@dataclass(frozen=True)
class Frame:
    fn: str
    file: str = ""  # source file, empty without debug info
    line: int = 0
    obj: str = ""  # shared object, when there is no source file

    def where(self):
        if self.file:
            return f"{os.path.basename(self.file)}:{self.line}"
        return os.path.basename(self.obj) if self.obj else ""

    def __str__(self):
        where = self.where()
        return f"{self.fn} ({where})" if where else self.fn


# C/C++ runtime frames between an allocation and the code asking for it, eg
#  vector (stl_vector.h:571) or allocate (new_allocator.h:151). Other libraries
#  (hdf5, mpi) are kept: their frames are often the answer
_STD_FILE = re.compile(
    r"^(stl_\w+\.h|new_allocator\.h|alloc_traits\.h|allocator\.h|basic_string\.\w+"
    r"|unique_ptr\.h|shared_ptr\w*\.h|hashtable\w*\.h|vector\.tcc|move\.h|ptr_traits\.h"
    r"|allocated_ptr\.h|stl_construct\.h|char_traits\.h|[a-z_]+)$"  # no extension: <vector>
)
_STD_FN = re.compile(r"^(std::|__gnu_cxx::|__cxx|operator new|__libc_|_dl_|_GLOBAL__)")
_STD_OBJ = re.compile(r"(libstdc\+\+|libc\+\+|libc\.so|libc-|ld-linux|vgpreload)")


def is_std(frame):
    """runtime/standard library frame, collapsed in reports"""
    if frame.file:
        return bool(
            _STD_FILE.match(os.path.basename(frame.file)) or _STD_FN.match(frame.fn)
        )
    if frame.obj:
        return bool(_STD_OBJ.search(frame.obj) or _STD_FN.match(frame.fn))
    return bool(_STD_FN.match(frame.fn)) or frame.fn in ("???", "")


def first_user_frame(frames):
    """first frame outside the C/C++ runtime and valgrind's own replacements"""
    for f in frames:
        if not is_std(f) and "vg_replace" not in f.file:
            return f
    return frames[0] if frames else None
