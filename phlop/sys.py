#
#
#
#
#


import os
import sys
from contextlib import contextmanager


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
