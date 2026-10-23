#ifndef SCAN_OUTPUT_H
#define SCAN_OUTPUT_H

// Scan output share: the scan coil voltages DE-Server outputs, for a process standing in
// for the microscope (the de-twin digital twin). In simulator mode there is no scan card;
// the scan simulator writes here what it would have written to the card's FIFO, block by
// block and as far ahead of the beam, so a client steering the scan sees the same latency
// it would on hardware.
//
// Shared memory "DE_ScanOutput":
//     ScanOutputHeader                 (kHeaderBytes reserved)
//     kCapacity ScanOutputPoint        a ring: point n is at n % kCapacity
//
// 1. At each scan start DE-Server writes the scan geometry, sets written = output = 0 and
//    then increments scanId.
// 2. It writes the voltages of point n into the ring, then sets written = n + 1. A block
//    is written at once, with the pixel offset read once for it, as on the card.
// 3. output counts the points the beam has reached. A frame of the camera covers
//    pointsPerFrame points from frame index x pointsPerFrame on.
//
// Points older than written - kCapacity have been overwritten. The Python side of this
// layout is de_twin/transport/scan_output.py.

#include <windows.h>
#include <cstdint>
#include <cstring>

namespace ScanOutput
{
	constexpr const char* kMappingName = "DE_ScanOutput";
	constexpr uint32_t    kMagic       = 0x4F534544;	// "DESO"
	constexpr uint32_t    kVersion     = 1;
	constexpr uint32_t    kHeaderBytes = 4096;
	constexpr uint64_t    kCapacity    = 1ull << 20;	// 16 MB of points, ~35 s at 30 kHz
}

struct ScanOutputHeader
{
	uint32_t magic;
	uint32_t version;
	uint64_t capacity;
	volatile uint64_t scanId;		// incremented after the geometry of a new scan is written
	volatile uint64_t written;		// voltages are valid for points [written - capacity, written)
	volatile uint64_t output;		// points the beam has reached
	uint64_t totalPoints;			// in the whole acquisition, all repeats
	uint32_t sizeX;					// Scan - Size X/Y: the full grid
	uint32_t sizeY;
	double   voltageMinH;			// Scan - Voltage H/V Min/Max: the range the grid spans
	double   voltageMaxH;
	double   voltageMinV;
	double   voltageMaxV;
	double   rotationDeg;
	uint32_t flipHorizontal;
	uint32_t flipVertical;
	double   pointsPerFrame;
	double   outputHz;				// scan points a second
};

struct ScanOutputPoint
{
	double h;
	double v;
};

class ScanOutputShare
{
public:
	~ScanOutputShare()
	{
		if (m_view)
			UnmapViewOfFile(m_view);
		if (m_mapping)
			CloseHandle(m_mapping);
	}

	// Create the mapping (once), write the geometry and start a new scan.
	bool Begin(const ScanOutputHeader& geometry)
	{
		if (!Open())
			return false;
		ScanOutputHeader* h = m_header;
		ScanOutputHeader next = geometry;
		const uint64_t scanId = h->scanId;
		next.scanId = scanId;	// changes only once the new geometry is in place
		std::memcpy(h, &next, sizeof(ScanOutputHeader));
		h->magic = ScanOutput::kMagic;
		h->version = ScanOutput::kVersion;
		h->capacity = ScanOutput::kCapacity;
		h->written = 0;
		h->output = 0;
		MemoryBarrier();
		InterlockedExchange64(reinterpret_cast<volatile LONG64*>(&h->scanId), static_cast<LONG64>(scanId + 1));
		return true;
	}

	// The next point's voltages; Publish makes everything written since visible.
	void Write(double h, double v)
	{
		ScanOutputPoint& p = m_points[m_pending % ScanOutput::kCapacity];
		p.h = h;
		p.v = v;
		m_pending++;
	}

	void Publish()
	{
		if (!m_header)
			return;
		MemoryBarrier();
		InterlockedExchange64(reinterpret_cast<volatile LONG64*>(&m_header->written), static_cast<LONG64>(m_pending));
	}

	void SetOutput(uint64_t points)
	{
		if (m_header)
			InterlockedExchange64(reinterpret_cast<volatile LONG64*>(&m_header->output), static_cast<LONG64>(points));
	}

	uint64_t Written() const { return m_pending; }
	bool IsOpen() const { return m_header != nullptr; }

private:
	bool Open()
	{
		m_pending = 0;
		if (m_header)
			return true;
		const uint64_t totalBytes = ScanOutput::kHeaderBytes + ScanOutput::kCapacity * sizeof(ScanOutputPoint);
		m_mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
			static_cast<DWORD>(totalBytes >> 32), static_cast<DWORD>(totalBytes), ScanOutput::kMappingName);
		if (!m_mapping)
			return false;
		m_view = static_cast<uint8_t*>(MapViewOfFile(m_mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0));
		if (!m_view)
			return false;
		m_header = reinterpret_cast<ScanOutputHeader*>(m_view);
		m_points = reinterpret_cast<ScanOutputPoint*>(m_view + ScanOutput::kHeaderBytes);
		return true;
	}

	HANDLE m_mapping = nullptr;
	uint8_t* m_view = nullptr;
	ScanOutputHeader* m_header = nullptr;
	ScanOutputPoint* m_points = nullptr;
	uint64_t m_pending = 0;
};

#endif
