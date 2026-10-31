# phlop/sys.py

import os
import sys
from contextlib import contextmanager
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path


@contextmanager
def extend_sys_path(paths):
    if isinstance(paths, str):
        paths = [paths]
    old_path = sys.path[:]
    sys.path.extend(paths)
    try:
        yield
    finally:
        sys.path = old_path


@contextmanager
def extend_env(**environ):
    old_env = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_env)


@contextmanager
def scoped_modules(roots=None):
    """Modules first imported inside this block from under any of `roots` are
    dropped from sys.modules on exit, so a later file can import its own module
    of the same name (e.g. a sibling `common.py`) rather than the cached one.
    `roots` is yielded and may be extended inside the block, for when the
    directory is only known once the import has been resolved.

    Compiled extensions are kept: they can't be re-initialised in-process."""
    roots = [] if roots is None else roots
    before = set(sys.modules)
    try:
        yield roots
    finally:
        dirs = [Path(r).resolve() for r in roots]
        for name in set(sys.modules) - before:
            file = getattr(sys.modules.get(name), "__file__", None)
            if not file or file.endswith(tuple(EXTENSION_SUFFIXES)):
                continue
            if any(Path(file).resolve().is_relative_to(d) for d in dirs):
                sys.modules.pop(name, None)
