A scan follows DE-Server's timing for coil changes: it runs a repeat at a time, and a pixel
offset (or voltage, rotation or flip) written mid-scan reaches the beam one
``Scan - Output Buffer (milliseconds)`` later, on a repeat boundary, as the new DE-Server
property queues it. The offset is clamped to half the scan's voltage range, inside the scan
card's +-10 V. ``get_gpu_frames()`` now maps the twin's own frames out of GPU memory, through
DE-Server's GPU frame sharing protocol, instead of deapi's simulated stand-in.

The shared-memory face keeps up with a fast camera: with ``reuse``, a full frame pool is a
library that is picked from rather than rendered into, re-rendered when the column moves, and
frames go to DE-Server a batch at a time. Its C++ reader yields instead of sleeping for the
first few ms of a wait. A 256x256 stream reaches about 27,000 frames a second, from 580.

In a scan, the shared-memory face follows DE-Server's scan coil voltages (``DE_ScanOutput``,
which DE-Server's scan simulator writes as a scan card would be written): each frame is
rendered where its point's voltages put the beam, so a pixel offset, an XY array or drift
correction moves the beam with DE-Server's output-buffer latency. A probe held on one point
is rendered ``pool_size`` times and then picked from (about 25,000 frames a second at 256x256).
