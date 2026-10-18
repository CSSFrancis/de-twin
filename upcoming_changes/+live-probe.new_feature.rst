A scan follows DE-Server's timing for coil changes: it runs a repeat at a time, and a pixel
offset (or voltage, rotation or flip) written mid-scan reaches the beam one
``Scan - Output Buffer (milliseconds)`` later, on a repeat boundary, as the new DE-Server
property queues it. The offset is clamped to half the scan's voltage range, inside the scan
card's +-10 V. ``get_gpu_frames()`` now maps the twin's own frames out of GPU memory, through
DE-Server's GPU frame sharing protocol, instead of deapi's simulated stand-in.
