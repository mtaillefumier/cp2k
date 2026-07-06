#!/usr/bin/env python3

# Shared, dependency-free logic for checking a TEST_DIRS requirement string
# (e.g. "libint mpiranks%2==0") against a build's feature flags and rank
# count. Used by do_regtest_dask.py and by the CTest integration
# (ctest_run_one.py) so this parser exists in exactly one place.

import operator
from typing import List

# Requirement strings like "mpiranks==4" or "mpiranks%2==0" come straight from
# the TEST_DIRS file. Several conditions can be combined with "||" meaning
# "or", e.g. "mpiranks==1||mpiranks%2==0". The functions below evaluate them
# without using Python's eval(), so the supported syntax is spelled out here
# explicitly: comparisons must be listed longest-first, so that e.g. ">=" is
# recognized before the shorter ">" would incorrectly match part of it.
_MPIRANKS_COMPARISONS = [
    ("==", operator.eq),
    ("!=", operator.ne),
    (">=", operator.ge),
    ("<=", operator.le),
    (">", operator.gt),
    ("<", operator.lt),
]


def _mpiranks_requirement_met(condition: str, mpiranks: int) -> bool:
    for alternative in condition.split("||"):
        if _single_mpiranks_condition_met(alternative, mpiranks):
            return True
    return False


def _single_mpiranks_condition_met(condition: str, mpiranks: int) -> bool:
    for operator_text, compare in _MPIRANKS_COMPARISONS:
        if operator_text in condition:
            left_text, right_text = condition.split(operator_text, 1)
            left_value = _evaluate_mpiranks_expression(left_text, mpiranks)
            right_value = int(right_text)
            return compare(left_value, right_value)
    raise ValueError(f"Cannot parse requirement: {condition}")


def _evaluate_mpiranks_expression(expression: str, mpiranks: int) -> int:
    # The only expressions seen in practice are "mpiranks" and "mpiranks%N".
    if "%" in expression:
        _, modulus_text = expression.split("%", 1)
        return mpiranks % int(modulus_text)
    return mpiranks


def requirements_satisfied(
    requirements: List[str], flags: List[str], mpiranks: int
) -> bool:
    for requirement in requirements:
        if "mpiranks" in requirement:
            if not _mpiranks_requirement_met(requirement, mpiranks):
                return False
        elif requirement.startswith("!"):
            if requirement[1:] in flags:
                return False
        elif requirement not in flags:
            return False
    return True


# EOF
