import contextlib
import os


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
    import os

    cwd = os.getcwd()
    os.chdir(new_cwd)
    try:
        yield
    finally:
        os.chdir(cwd)
