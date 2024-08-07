#
#
#
#
#


import atexit
import subprocess

from phlop.logger import getLogger
from phlop.proc import run
from phlop.string import decode_bytes

logger = getLogger(__name__)


def current_branch():
    return decode_bytes(run("git branch --show-current").stdout)


def hashes(N=1):
    return decode_bytes(run(f"git log -{N} --pretty=format:%h").stdout).splitlines()


def log(N=10, use_short=False):
    short = '--pretty=format:"%h%x09%an%x09%ad%x09%s"' if use_short else ""
    return decode_bytes(run(f"git log -{N} {short}").stdout)


def branch_exists(branch):
    try:
        run(f"git show-branch {branch}", check=True)
    except subprocess.CalledProcessError:
        return False  # exit failure means branch does not exist
    return True


def checkout(branch, create=False):
    if create and not branch_exists(branch):
        run(f"git checkout -b {branch}", check=True)
    else:
        run(f"git checkout {branch}", check=True)


def git_branch_reset_at_exit():
    current_git_branch = current_branch()

    @atexit.register
    def _reset_():
        checkout(current_git_branch)
