#!/usr/bin/env python3
# check if parser and qbe backend include all types
# pylint: disable=import-error, wrong-import-position
import inspect
import pathlib
import sys
from collections import deque

sys.path.insert(
    0, str(pathlib.Path(__file__).parents[1].joinpath("celestine").resolve())
)

from parse import primitive_types as parser_types_map
from qbe import QBE
from types_info import Pointer, PrimitiveType

FAILURE = False


##### Gather types that need implementation
primitive_types = set()
incomplete_types = deque([PrimitiveType])

while len(incomplete_types) > 0:
    typ = incomplete_types.popleft()
    subclasses = typ.__subclasses__()
    for subclass in subclasses:
        # ignore abstract type classes:
        if not inspect.isabstract(subclass):
            primitive_types.add(subclass)
        incomplete_types.append(subclass)


for typ in primitive_types:
    # Test if QBE backend has implemented the type
    impl = QBE.get_type_impl(typ.__new__(typ))
    if impl is None:
        FAILURE = True
        print(f"QBE doesn't implement {typ}")

    # Check if parser has implemented the type
    if  (typ!=Pointer) and (typ not in parser_types_map.values()) :
        FAILURE = True
        print(
            f"Parser doesn't implement {typ}"
            "as it's not included in parser.py: primitive_types"
        )

if FAILURE:
    sys.exit(1)
