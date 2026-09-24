#ifndef EXTERNAL_FRAME_SOURCE_H
#define EXTERNAL_FRAME_SOURCE_H

// External frame source: camera frames produced by another process (for example the
// de-twin digital twin) and handed to GrabberSim through named shared memory.
// Selected in DE-Server with the test pattern "External Frame Source (Shared Memory)".
//
// Shared memory "DE_ExternalFrames":
//     ExternalFrameHeader              (kHeaderBytes reserved)
//     kSlotCount slots, each:          ExternalFrameSlot (64 bytes) + one frame of pixels
//
// 1. DE-Server creates the mapping. Before each acquisition it writes the request
//    (frame size, exposure mode, frame time, ...) and increments requestId.
// 2. The producer writes frame n into slot n % kSlotCount, sets slot.frameNumber = n + 1
//    last, then sets writeCount = n + 1. It never runs more than kSlotCount frames
//    ahead of readCount.
// 3. DE-Server copies frames in order and sets readCount = n + 1. Frames made for an
//    older request are skipped.
//
// Pixels are unsigned, row-major, frameWidth x frameHeight (the hw_frame, after hardware
// ROI and binning). The Python side of this layout is de_twin/transport/shm_layout.py.

#include <windows.h>
#include <cstddef>
#include <cstdint>

namespace ExternalFrames
{
	constexpr const char* kTestPattern     = "External Frame Source (Shared Memory)";
	constexpr const char* kMappingName     = "DE_ExternalFrames";
	constexpr uint32_t    kMagic           = 0x57544544;	// "DETW"
	constexpr uint32_t    kVersion         = 2;
	constexpr uint32_t    kSlotCount       = 8;
	constexpr uint32_t    kHeaderBytes     = 4096;
	constexpr uint32_t    kSlotHeaderBytes = 64;
}

// What DE-Server asks the producer for, per acquisition
struct ExternalFrameRequest
{
	uint32_t exposureMode;          // ExposureMode: 0 dark, 1 trial, 2 gain, 3 normal, ...
	uint32_t frameWidth;            // hw_frame (after hardware ROI and binning)
	uint32_t frameHeight;
	uint32_t bytesPerPixel;         // 1 or 2
	uint32_t sensorWidth;
	uint32_t sensorHeight;
	uint32_t roiX, roiY, roiWidth, roiHeight;
	uint32_t binX, binY;
	uint32_t scanWidth, scanHeight; // 0 when not scanning
	uint32_t framesPerScanPoint;
	uint32_t reserved;
	uint64_t frameTimeNs;
	uint64_t totalFrames;           // 0 = until the acquisition stops
};

struct ExternalFrameHeader
{
	uint32_t             magic;          // written once, when DE-Server creates the mapping
	uint32_t             version;
	uint32_t             slotCount;
	uint32_t             reserved;
	uint64_t             slotBytes;      // slot header + largest frame
	uint64_t             requestId;      // incremented by DE-Server for every acquisition
	uint32_t             acquiring;      // 1 while DE-Server wants frames
	uint32_t             reserved2;
	ExternalFrameRequest request;
	uint64_t             readCount;      // frames consumed (DE-Server)
	uint64_t             writeCount;     // frames published (producer)
};

struct ExternalFrameSlot
{
	uint64_t frameNumber;           // global frame number + 1, written last: the slot is ready
	uint64_t requestId;             // the request this frame was made for
	uint64_t frameIndex;            // index within the acquisition
	uint32_t width;
	uint32_t height;
	uint32_t bytesPerPixel;
	uint32_t flags;                 // bit 0: beam was blanked (dark frame)
	uint8_t  reserved[24];
};

static_assert(sizeof(ExternalFrameRequest) == 80, "ExternalFrameRequest layout");
static_assert(offsetof(ExternalFrameHeader, request) == 40, "ExternalFrameHeader layout");
static_assert(offsetof(ExternalFrameHeader, readCount) == 120, "ExternalFrameHeader layout");
static_assert(offsetof(ExternalFrameHeader, writeCount) == 128, "ExternalFrameHeader layout");
static_assert(sizeof(ExternalFrameSlot) == ExternalFrames::kSlotHeaderBytes, "ExternalFrameSlot layout");

class ExternalFrameSource
{
public:
	explicit ExternalFrameSource(const char* mappingName = ExternalFrames::kMappingName) : m_name(mappingName) {}
	~ExternalFrameSource();

	// Publish the request for a new acquisition (creates the shared memory the first time)
	bool Begin(const ExternalFrameRequest& request);

	// Copy the next frame into dst (frameHeight rows of frameWidth pixels, dstStrideBytes apart).
	// Returns false if no frame arrived within timeoutMs.
	bool ReadFrame(void* dst, size_t dstStrideBytes, uint32_t timeoutMs);

	// The acquisition is over
	void End();

private:
	bool Open(uint64_t maxFrameBytes);
	ExternalFrameSlot* Slot(uint64_t frame) const;

	const char*          m_name;
	HANDLE               m_mapping   = nullptr;
	uint8_t*             m_view      = nullptr;
	ExternalFrameHeader* m_header    = nullptr;
	uint64_t             m_requestId = 0;
};

#endif // EXTERNAL_FRAME_SOURCE_H
