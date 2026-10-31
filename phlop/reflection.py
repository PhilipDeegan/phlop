# phlop/reflection.py

import importlib
import importlib.util
import inspect
import logging
import os
import weakref
from pathlib import Path

from phlop.sys import extend_sys_path, scoped_modules

logger = logging.getLogger(__name__)

FORCE_RAISE_ON_IMPORT_ERROR = os.getenv(
    "PHLOP_FORCE_RAISE_ON_IMPORT_ERROR", "False"
).lower() in ("true", "1", "t")

# source file of each class returned by classes_in_file, recorded while its
# module is still imported - the module itself is dropped from sys.modules
_CLASS_FILES = weakref.WeakKeyDictionary()


def class_file(cls):
    """Source file `cls` was defined in"""
    return _CLASS_FILES.get(cls) or inspect.getfile(cls)


def classes_in_file(file_path, subclasses_only=None, fail_on_import_error=True):
    file = Path(file_path)
    module = str(file).replace(os.path.sep, ".")[:-3]
    assert module

    if subclasses_only is not None and not isinstance(subclasses_only, list):
        subclasses_only = [subclasses_only]

    classes = []

    with (
        extend_sys_path([os.getcwd(), str(file.parent)]),
        scoped_modules() as roots,
    ):
        try:
            # where `module` resolves (via sys.path), not necessarily file.parent
            spec = importlib.util.find_spec(module)
            if spec and spec.origin:
                roots.append(Path(spec.origin).parent)
            mod = importlib.import_module(module)
            for name, cls in inspect.getmembers(mod, inspect.isclass):
                if cls.__module__ != mod.__name__:
                    continue  # imported, belongs to the file that defines it
                should_add = subclasses_only is None or any(
                    issubclass(cls, sub) for sub in subclasses_only
                )
                if should_add:
                    _CLASS_FILES[cls] = inspect.getfile(cls)
                    classes += [cls]
        except (ValueError, ModuleNotFoundError) as e:
            if fail_on_import_error or FORCE_RAISE_ON_IMPORT_ERROR:
                raise
            logger.error(f"Skipping on error: {e} in module {module}")

    return classes


def classes_in_directory(path, subclasses_only=None):
    classes = []
    for file in list(Path(path).glob("**/*.py")):
        classes += classes_in_file(file, subclasses_only)
    return classes
