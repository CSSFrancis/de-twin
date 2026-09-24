The deapi face treats ``start_acquisition(0)`` as live view (repeat until stopped), as
DE-Server does, instead of a single acquisition; and repeated acquisitions no longer repeat
the same detector noise, because noise is now seeded by a twin-wide frame counter rather than
the frame index within each request.
