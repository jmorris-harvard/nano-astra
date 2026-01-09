#include "astra-sim/workload/JBasicCommunicationStatistics.hh"

#include <iostream>

using namespace Jalil;
using namespace AstraSim;
using namespace std;

CommunicationBlock::CommunicationBlock (uint64_t start, uint64_t end, uint64_t tsize, uint64_t psize, uint64_t asize, uint64_t rsize) : _start (start), _end (end),
     _tsize (tsize), _psize (psize), _asize (asize), _rsize (rsize) {}

const vector<StatisticsType> BasicCommunicationStatistics::_targets = {
  StatisticsType::Communication  
};

BasicCommunicationStatistics::BasicCommunicationStatistics (void)
  : StatisticsProcessor (),
    _aggTotalBytesSent (0),
    _aggTransmitTime (0),
    _totalTransmitEvents (0),
    _totalTSize (0),
    _totalPSize (0),
    _totalASize (0),
    _totalRSize (0) {}

void BasicCommunicationStatistics::add (BasicEventHandlerData *ehd) {
  this->addInternal ((SendPacketEventHandlerData *) ehd);
}

void BasicCommunicationStatistics::addInternal (SendPacketEventHandlerData *sehd) {
  uint64_t start, end;
  uint64_t tsize = 0, psize = 0, asize = 0, rsize = 0;
  uint32_t i;
  auto packetShadows = sehd->_packetTracker.readPackets ();
  for (i = 0; i < packetShadows.size (); ++i) {
    if (i == 0) {
      start = packetShadows[i]._timestamp;
    } else if (i == packetShadows.size () - 1) {
      end = packetShadows[i]._timestamp;
    }
    if (packetShadows[i]._loc == PacketLocation::TRANSMIT) {
      tsize = tsize + packetShadows[i]._size;
    } else if (packetShadows[i]._loc == PacketLocation::PROPAGATE) {
      psize = psize + packetShadows[i]._size;
    } else if (packetShadows[i]._loc == PacketLocation::ARRIVE) {
      asize = asize + packetShadows[i]._size;
    } else if (packetShadows[i]._loc == PacketLocation::RECEIVE) {
      rsize = rsize + packetShadows[i]._size;
    } else {
      cout << static_cast<uint64_t> (packetShadows[i]._type) << " " << static_cast<uint64_t> (packetShadows[i]._loc) << " " << packetShadows[i]._node << " " << packetShadows[i]._port << endl;
    }

    pair<unsigned, unsigned> key (packetShadows[i]._node, packetShadows[i]._port);
    if (this->_portUsage.find (key) == this->_portUsage.end ()) {
      this->_portUsage[key] = 0;
    }
    this->_portUsage[key] = this->_portUsage[key] + 1;
  }
  this->_blocks.emplace_back (start, end, tsize, psize, asize, rsize);
}

void BasicCommunicationStatistics::process (void) {
  for (auto &block : _blocks) {
    this->_aggTotalBytesSent = this->_aggTotalBytesSent + block._tsize;
    this->_aggTransmitTime = this->_aggTransmitTime + (block._end - block._start);
    this->_totalTransmitEvents = this->_totalTransmitEvents + 1;
    this->_totalTSize = this->_totalTSize + block._tsize;
    this->_totalPSize = this->_totalPSize + block._psize;
    this->_totalASize = this->_totalASize + block._asize;
    this->_totalRSize = this->_totalRSize + block._rsize;
  }
  this->_blocks.clear ();
}

void BasicCommunicationStatistics::report (void) {
  cout << "Total Data Transferred: " << this->_aggTotalBytesSent << endl;
  cout << "Average Transmission Time: " << (double) this->_aggTransmitTime / (double) this->_totalTransmitEvents << endl;
  cout << "Total Transmit Size: " << this->_totalTSize << endl;
  cout << "Total Propagate Size: " << this->_totalPSize << endl;
  cout << "Total Arrive Size: " << this->_totalASize << endl;
  cout << "Total Receive Size: " << this->_totalRSize << endl;
  cout << "Port Usage Summary:" << endl;
  for (const auto &[key, value] : this->_portUsage) {
    unsigned node = key.first;
    unsigned port = key.second;
    cout << "[ node (" << node << "), port (" << port << ") ] = " << value << endl;
  }
}

const vector<StatisticsType> &BasicCommunicationStatistics::targets (void) const {
  return _targets;
}
