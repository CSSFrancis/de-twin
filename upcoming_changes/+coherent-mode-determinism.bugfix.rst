Coherent 4D-STEM data no longer depends on the machine: the probe's partial-coherence
modes were truncated through a degenerate pair of eigenmodes, whose basis LAPACK chooses
differently with BLAS threading, so the same request gave different patterns on different
computers. Truncation now keeps degenerate groups whole.
