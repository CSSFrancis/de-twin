#include "ExternalFrameSource.h"

#include <algorithm>
#include <cstring>

ExternalFrameSource::~ExternalFrameSource()
{
	End();
	if (m_view)
		UnmapViewOfFile(m_view);
	if (m_mapping)
		CloseHandle(m_mapping);
}

bool ExternalFrameSource::Open(uint64_t maxFrameBytes)
{
	if (m_header && m_header->slotBytes >= ExternalFrames::kSlotHeaderBytes + maxFrameBytes)
		return true;
	if (m_header)
		return false;	// the mapping already exists with smaller slots; restart both processes

	const uint64_t slotBytes = (ExternalFrames::kSlotHeaderBytes + maxFrameBytes + 63) / 64 * 64;
	const uint64_t totalBytes = ExternalFrames::kHeaderBytes + ExternalFrames::kSlotCount * slotBytes;
	m_mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
		static_cast<DWORD>(totalBytes >> 32), static_cast<DWORD>(totalBytes), m_name);
	if (!m_mapping)
		return false;
	const bool existed = GetLastError() == ERROR_ALREADY_EXISTS;

	m_view = static_cast<uint8_t*>(MapViewOfFile(m_mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0));
	if (!m_view)
		return false;
	m_header = reinterpret_cast<ExternalFrameHeader*>(m_view);

	if (existed && m_header->magic == ExternalFrames::kMagic)
		return m_header->version == ExternalFrames::kVersion && m_header->slotBytes >= ExternalFrames::kSlotHeaderBytes + maxFrameBytes;

	std::memset(m_view, 0, ExternalFrames::kHeaderBytes);
	m_header->version   = ExternalFrames::kVersion;
	m_header->slotCount = ExternalFrames::kSlotCount;
	m_header->slotBytes = slotBytes;
	MemoryBarrier();
	m_header->magic     = ExternalFrames::kMagic;
	return true;
}

ExternalFrameSlot* ExternalFrameSource::Slot(uint64_t frame) const
{
	return reinterpret_cast<ExternalFrameSlot*>(m_view + ExternalFrames::kHeaderBytes + (frame % m_header->slotCount) * m_header->slotBytes);
}

bool ExternalFrameSource::Begin(const ExternalFrameRequest& request)
{
	const uint64_t frameBytes = static_cast<uint64_t>(request.frameWidth) * request.frameHeight * request.bytesPerPixel;
	const uint64_t sensorBytes = static_cast<uint64_t>(request.sensorWidth) * request.sensorHeight * 2;
	if (!Open((std::max)(frameBytes, sensorBytes)) || m_header->slotBytes < ExternalFrames::kSlotHeaderBytes + frameBytes)
		return false;

	m_header->request = request;
	m_header->readCount = m_header->writeCount;	// anything still in the ring belongs to an older request
	m_header->acquiring = 1;
	MemoryBarrier();
	m_requestId = static_cast<uint64_t>(InterlockedIncrement64(reinterpret_cast<volatile LONG64*>(&m_header->requestId)));
	return true;
}

bool ExternalFrameSource::ReadFrame(void* dst, size_t dstStrideBytes, uint32_t timeoutMs)
{
	if (!m_header || m_requestId == 0)
		return false;

	const ULONGLONG deadline = GetTickCount64() + timeoutMs;
	while (true)
	{
		const uint64_t n = m_header->readCount;
		ExternalFrameSlot* slot = Slot(n);
		if (static_cast<uint64_t>(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64*>(&slot->frameNumber), 0, 0)) == n + 1)
		{
			const bool current = slot->requestId == m_requestId;
			if (current)
			{
				const ExternalFrameRequest& r = m_header->request;
				const size_t rowBytes = static_cast<size_t>((std::min)(slot->width, r.frameWidth)) * r.bytesPerPixel;
				const uint32_t rows = (std::min)(slot->height, r.frameHeight);
				const uint8_t* src = reinterpret_cast<const uint8_t*>(slot) + ExternalFrames::kSlotHeaderBytes;
				for (uint32_t y = 0; y < rows; y++)
					std::memcpy(static_cast<uint8_t*>(dst) + y * dstStrideBytes, src + static_cast<size_t>(y) * slot->width * slot->bytesPerPixel, rowBytes);
			}
			InterlockedExchange64(reinterpret_cast<volatile LONG64*>(&m_header->readCount), static_cast<LONG64>(n + 1));
			if (current)
				return true;
			continue;	// a leftover frame from an older request
		}
		if (GetTickCount64() >= deadline)
			return false;
		Sleep(1);
	}
}

void ExternalFrameSource::End()
{
	if (m_header)
		m_header->acquiring = 0;
}
