import importlib
import inspect
import os
from pathlib import Path


def classes_in_file(file, subclasses_only=None):
    module = str(file).replace(os.path.sep, ".")[:-3]

    if subclasses_only is not None and not isinstance(subclasses_only, list):
        subclasses_only = [subclasses_only]

    classes = []
    for name, cls in inspect.getmembers(
        importlib.import_module(module), inspect.isclass
    ):
        should_add = subclasses_only == None or any(
            [issubclass(cls, sub) for sub in subclasses_only]
        )
        if should_add:
            classes += [cls]
    return classes


def classes_in_directory(path, subclasses_only=None):
    classes = []
    for file in list(Path(path).glob("**/*.py")):
        classes += classes_in_file(file, subclasses_only)
    return classes
