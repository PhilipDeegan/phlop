#
#
#
#
#


import contextlib
import os
import platform
from pathlib import Path


def scan_dir(path, files_only=False, dirs_only=False, drop=[]):
    assert os.path.exists(path)
    checks = [
        lambda entry: not files_only or (files_only and entry.is_file()),
        lambda entry: not dirs_only or (dirs_only and entry.is_dir()),
        lambda entry: entry.name not in drop,
    ]
    return [
        entry.name
        for entry in os.scandir(path)
        if all([check(entry) for check in checks])
    ]


@contextlib.contextmanager
def pushd(new_cwd):
    if not os.path.exists(new_cwd):
        raise RuntimeError("phlop.os.pushd: new_cwd does not exist")

    cwd = os.getcwd()
    os.chdir(new_cwd)
    try:
        yield
    finally:
        os.chdir(cwd)


def write_to_file(file, contents, mode="w", skip_if_empty=True):
    if contents or not skip_if_empty:
        try:
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            with open(file, mode) as f:
                f.write(contents)
        except IOError as e:
            raise RuntimeError(f"Failed to write to file {file}: {e}")


def env_sep():
    return ";" if any(platform.win32_ver()) else ":"
