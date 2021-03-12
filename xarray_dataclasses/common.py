# standard library
from dataclasses import Field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple


# third-party packages
from typing_extensions import Protocol


# submodules
from .typing import is_attr, is_coord, is_data, is_name


# type hints
class DataClass(Protocol):
    """Type hint for dataclass and its instance."""

    __dataclass_fields__: Dict[str, Field]


# helper functions (internal)
def _gen_fields(
    inst: DataClass, type_filter: Optional[Callable[..., bool]] = None
) -> Iterable[Tuple[Field, Any]]:
    """Generate field-value pairs from a dataclass instance.

    Args:
        inst: An instance of dataclass.
        type_filter: If specified, only field-value pairs
            s.t. ``type_filter(field.type) == True`` are yielded.

    Yields:
        Field-value pairs as tuple.

    """
    for name, field in inst.__dataclass_fields__.items():
        if type_filter is None or type_filter(field.type):
            yield field, getattr(inst, name)


def _get_one(obj: Mapping) -> Any:
    """Return value of mapping if it has an exactly one entry."""
    if len(obj) != 1:
        raise ValueError("obj must have an exactly one entry.")

    return next(iter(obj.values()))
