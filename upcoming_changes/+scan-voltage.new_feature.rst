STEM scans now follow DE-Server's scan coil voltages: ``Scan - Voltage H/V Min/Max`` is the
field the full scan grid spans (a smaller range zooms in, moving it pans), and
``Scan - Pixel Offset X/Y`` moves the probe by whole or fractional scan points. ``Scan - Rotation``
and ``Scan - Flip Horizontally/Vertically`` turn and mirror the grid. A live scan
(``start_acquisition(0)``) picks up offset, voltage, rotation and flip changes at its next
repetition, so a client can move the probe, correct drift, or pan and zoom while it scans.
