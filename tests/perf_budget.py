"""Timing budgets for the performance smoke tests.

The budgets are tuned on a multi-core workstation. Shared CI runners have 2-4 cores, so
CI sets ``DE_TWIN_PERF_SLACK`` to scale every budget rather than skipping the checks: a
large regression still fails there.
"""

import os

SLACK = float(os.environ.get("DE_TWIN_PERF_SLACK", "1"))


def budget(seconds: float) -> float:
    return seconds * SLACK
