#
#
#
#
#

import importlib
import inspect
import logging
import os
from pathlib import Path

from phlop.sys import extend_sys_path


def classes_in_file(file, subclasses_only=None, fail_on_import_error=True):
    assert file
    module = str(file).replace(os.path.sep, ".")[:-3]
    assert module

    if subclasses_only is not None and not isinstance(subclasses_only, list):
        subclasses_only = [subclasses_only]

    classes = []

    with extend_sys_path(os.getcwd()):
        try:
            for name, cls in inspect.getmembers(
                importlib.import_module(module), inspect.isclass
            ):
                should_add = subclasses_only == None or any(
                    [issubclass(cls, sub) for sub in subclasses_only]
                )
                if should_add:
                    classes += [cls]
        except (ValueError, ModuleNotFoundError) as e:
            if fail_on_import_error:
                raise e
            logging.error(f"Skipping on error: {e} in module {module}")

    return classes


def classes_in_directory(path, subclasses_only=None):
    classes = []
    for file in list(Path(path).glob("**/*.py")):
        classes += classes_in_file(file, subclasses_only)
    return classes
