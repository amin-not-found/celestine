#!/usr/bin/env python3
# check if parser and qbe backend include all types
# pylint: disable=import-error, wrong-import-position
import sys
import pathlib
from collections import deque

sys.path.insert(
    0, str(pathlib.Path(__file__).parents[1].joinpath("celestine").resolve())
)

from type import PrimitiveType
from parse import primitive_types as parser_types_map
from gen import QBE


FAILURE = False

primitive_types = set()
incomplete_types = deque([PrimitiveType])

while len(incomplete_types) > 0:
    typ = incomplete_types.popleft()
    for subclass in typ.__subclasses__():
        if (
            hasattr(subclass, "final_type")
            and subclass.final_type
            # ignore type implementations:
            and not any(base in primitive_types for base in subclass.__bases__)
        ):
            primitive_types.add(subclass)
            continue
        incomplete_types.append(subclass)


parser_types = parser_types_map.values()

for t in primitive_types:
    if not t in QBE.type_implementations:
        FAILURE = True
        print(f"QBE doesn't implement {t}")

    if not t in parser_types:
        FAILURE = True
        print(
            f"Parser doesn't implement {t}"
            "as it's not included in parser.py:primitive_types"
        )

if FAILURE:
    sys.exit(1)
