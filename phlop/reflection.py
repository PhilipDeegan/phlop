import importlib
import inspect
import logging
import os
import sys
from pathlib import Path


def classes_in_file(file, subclasses_only=None, fail_on_import_error=True):
    module = str(file).replace(os.path.sep, ".")[:-3]

    if subclasses_only is not None and not isinstance(subclasses_only, list):
        subclasses_only = [subclasses_only]

    classes = []
    old_path = sys.path
    try:
        sys.path += [os.getcwd()]
        for name, cls in inspect.getmembers(
            importlib.import_module(module), inspect.isclass
        ):
            should_add = subclasses_only == None or any(
                [issubclass(cls, sub) for sub in subclasses_only]
            )
            if should_add:
                classes += [cls]
    except ModuleNotFoundError as e:
        if fail_on_import_error:
            raise e
        logging.error(f"Skipping on error: {e} in module {module}")
    finally:
        sys.path = old_path

    return classes


def classes_in_directory(path, subclasses_only=None):
    classes = []
    for file in list(Path(path).glob("**/*.py")):
        classes += classes_in_file(file, subclasses_only)
    return classes
