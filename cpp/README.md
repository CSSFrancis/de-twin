# DE-Server external frame source

`ExternalFrameSource.h/.cpp` let DE-Server's `GrabberSim` take camera frames from another
process through shared memory. The de-twin digital twin (`de-twin serve --shm`) is the
intended producer. Frames then go through the normal DE-Server pipeline exactly as
simulator frames do: frame-number stamp, `RevertToCrude`, `CheckBuffer`, references,
counting, saving.

## What changes in DE-Server

| File | Change |
|---|---|
| `Grabber/ExternalFrameSource.h/.cpp` | new, about 220 lines including the layout comments |
| `Grabber/GrabberSim.h` | include, and one `ExternalFrameSource m_externalFrames` member |
| `Grabber/GrabberSim.cpp` | `MakeExternalFrameRequest()`; `Begin()` when an acquisition starts; a `GenerateFrame` branch that reads frames; `End()` when the acquisition finishes; no frame-time benchmark for this pattern |
| `Camera/Camera.cpp` | adds the test pattern name to the list |
| `CameraServer.vcxproj(.filters)` | the two new files |

## How it works

1. When an acquisition starts with the test pattern **External Frame Source (Shared Memory)**,
   GrabberSim creates the shared memory `DE_ExternalFrames` (first time only). It then writes
   the request: `hw_frame` size, bytes per pixel, sensor, hardware ROI and binning, frame
   time, number of frames (0 = live), exposure mode and scan size. Finally it increments
   `requestId`.
2. The producer fills slots with frames. For each frame of a buffer, GrabberSim calls
   `ReadFrame`, which waits for the next frame and copies it into the grab buffer. If no
   frame arrives within 2 x frame time + 1 s, the frame is left blank and a warning is
   logged, so acquisitions never hang.
3. When the acquisition ends, `acquiring` goes to 0, and the producer stops.

The layout is documented at the top of `ExternalFrameSource.h`. Its Python mirror is
`de_twin/transport/shm_layout.py`, and `tests/test_shm_layout.py` checks the two agree
against the compiled header.

## Running it

1. In `configurations/server.xml`, set `GrabberType="Software"`. Any `Camera Type` works;
   it sets the sensor size, topology and bit depth.
2. Start the twin: `de-twin serve --shm --soap 5002 --camera DE16`. The `--soap` flag lets
   DE-Server poll the twin's microscope as its TEM-Channel: set
   *Instrument Client Address* to that machine.
3. In DE-MC, pick the test pattern **External Frame Source (Shared Memory)** and acquire.

Dark references work because the twin blanks the beam when the request's exposure mode
is Dark.

## Scan coil voltages

`ScanOutput.h` (DE-Server's `Scanner/ScanOutput.h`) is the scan card's side. In simulator
mode there is no card, so the scan simulator writes the voltages it would have sent into
the shared memory `DE_ScanOutput`: block by block, with the pixel offset read once per
block, and as far ahead of the beam as the card's FIFO would hold
(`Scan - Output Buffer (milliseconds)`). The twin renders each frame where its point's
voltages put the beam, so a pixel offset, an XY array or drift correction steers it with
the latency a real scan card gives.

| File | Change |
|---|---|
| `Scanner/ScanOutput.h` | new: the layout and a small writer |
| `Scanner/ScanController.h/.cpp` | `_SizeOutputBlocks()` (the card's block sizing, now shared with the simulator), `_BeginSimulatedOutput()` when a simulated scan starts, `_WriteSimulatedOutputBlock()` as the simulator advances |

Its Python mirror is `de_twin/transport/scan_output.py`, checked against the compiled
header by `tests/test_scan_output.py`.

## Testing without DE-Server

`build.bat` builds `build\test_consumer.exe` from the same `ExternalFrameSource.cpp`:

* `test_consumer layout` prints every struct offset, compared by `tests/test_shm_layout.py`
  and `tests/test_scan_output.py`.
* `test_consumer <mapping> <frames> <w> <h> <exposureMode>` acts like DE-Server. The
  twin's shared-memory face serves it in `tests/test_shm_interop.py`.
