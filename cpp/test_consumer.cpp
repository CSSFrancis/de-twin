// Stand-alone check of ExternalFrameSource, used by tests/test_shm_interop.py.
//
//   test_consumer layout
//       print every field offset, to compare with de_twin/transport/shm_layout.py
//   test_consumer <mapping> <frames> <width> <height> <exposureMode>
//       act like DE-Server: publish a request, read the frames, print one line per frame

#include "ExternalFrameSource.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define FIELD(T, f) std::printf("%s.%s %zu\n", #T, #f, offsetof(T, f))

int main(int argc, char** argv)
{
	if (argc == 2 && std::strcmp(argv[1], "layout") == 0)
	{
		FIELD(ExternalFrameHeader, magic);
		FIELD(ExternalFrameHeader, version);
		FIELD(ExternalFrameHeader, slotCount);
		FIELD(ExternalFrameHeader, slotBytes);
		FIELD(ExternalFrameHeader, requestId);
		FIELD(ExternalFrameHeader, acquiring);
		FIELD(ExternalFrameHeader, request);
		FIELD(ExternalFrameHeader, readCount);
		FIELD(ExternalFrameHeader, writeCount);
		FIELD(ExternalFrameRequest, exposureMode);
		FIELD(ExternalFrameRequest, frameWidth);
		FIELD(ExternalFrameRequest, frameHeight);
		FIELD(ExternalFrameRequest, bytesPerPixel);
		FIELD(ExternalFrameRequest, sensorWidth);
		FIELD(ExternalFrameRequest, sensorHeight);
		FIELD(ExternalFrameRequest, roiX);
		FIELD(ExternalFrameRequest, roiY);
		FIELD(ExternalFrameRequest, roiWidth);
		FIELD(ExternalFrameRequest, roiHeight);
		FIELD(ExternalFrameRequest, binX);
		FIELD(ExternalFrameRequest, binY);
		FIELD(ExternalFrameRequest, scanWidth);
		FIELD(ExternalFrameRequest, scanHeight);
		FIELD(ExternalFrameRequest, framesPerScanPoint);
		FIELD(ExternalFrameRequest, frameTimeNs);
		FIELD(ExternalFrameRequest, totalFrames);
		FIELD(ExternalFrameSlot, frameNumber);
		FIELD(ExternalFrameSlot, requestId);
		FIELD(ExternalFrameSlot, frameIndex);
		FIELD(ExternalFrameSlot, width);
		FIELD(ExternalFrameSlot, height);
		FIELD(ExternalFrameSlot, bytesPerPixel);
		FIELD(ExternalFrameSlot, flags);
		return 0;
	}
	if (argc != 6)
	{
		std::printf("usage: test_consumer layout | <mapping> <frames> <width> <height> <exposureMode>\n");
		return 2;
	}

	const int frames = std::atoi(argv[2]);
	ExternalFrameRequest r = {};
	r.frameWidth = static_cast<uint32_t>(std::atoi(argv[3]));
	r.frameHeight = static_cast<uint32_t>(std::atoi(argv[4]));
	r.exposureMode = static_cast<uint32_t>(std::atoi(argv[5]));
	r.bytesPerPixel = 2;
	r.sensorWidth = r.roiWidth = r.frameWidth;
	r.sensorHeight = r.roiHeight = r.frameHeight;
	r.binX = r.binY = 1;
	r.frameTimeNs = 10000000;
	r.totalFrames = static_cast<uint64_t>(frames);

	ExternalFrameSource source(argv[1]);
	if (!source.Begin(r))
	{
		std::printf("RESULT cannot open shared memory\n");
		return 1;
	}
	std::printf("READY\n");
	std::fflush(stdout);

	std::vector<uint16_t> frame(static_cast<size_t>(r.frameWidth) * r.frameHeight);
	for (int i = 0; i < frames; i++)
	{
		if (!source.ReadFrame(frame.data(), r.frameWidth * sizeof(uint16_t), 20000))
		{
			std::printf("RESULT timeout at frame %d\n", i);
			return 1;
		}
		unsigned long long sum = 0;
		for (uint16_t v : frame)
			sum += v;
		std::printf("FRAME i=%d sum=%llu first=%u last=%u\n", i, sum, frame.front(), frame.back());
	}
	source.End();
	std::printf("RESULT ok\n");
	return 0;
}
