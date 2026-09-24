Raised the minimum supported versions to numpy 2.0, scipy 1.13 and diffpy.structure 3.2
(and matplotlib 3.9 for the docs), the oldest set the test suite passes with. Timing
budgets in the performance tests now scale with ``DE_TWIN_PERF_SLACK`` so shared CI
runners can check them without being tuned to one workstation.
