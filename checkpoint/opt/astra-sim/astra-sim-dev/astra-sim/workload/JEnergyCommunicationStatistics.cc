#include "astra-sim/workload/JEnergyCommunicationStatistics.hh"

#include <iostream>
#include <cstdlib>

using namespace Jalil;
using namespace AstraSim;
using namespace std;
using json = nlohmann::json;

EnergyCommunicationBlock::EnergyCommunicationBlock (uint64_t start, uint64_t end, uint64_t tsize, uint64_t psize, uint64_t asize, uint64_t rsize) : _start (start), _end (end),
     _tsize (tsize), _psize (psize), _asize (asize), _rsize (rsize) {}

EnergyCommunicationConfig::EnergyCommunicationConfig (double pJPerBitTx, double pJPerBitRx)
  : _pJPerBitTx (pJPerBitTx), _pJPerBitRx (pJPerBitRx) {}

const vector<StatisticsType> EnergyCommunicationStatistics::_targets = {
  StatisticsType::Communication  
};

EnergyCommunicationStatistics::EnergyCommunicationStatistics (json config)
  : StatisticsProcessor (),
    _aggTotalBytesSent (0),
    _aggTransmitTime (0),
    _totalTransmitEvents (0),
    _totalTSize (0),
    _totalPSize (0),
    _totalASize (0),
    _totalRSize (0),
    _config (0.0, 0.0) {
  if (!config.contains ("tx")) {
    cerr << "energy communication statistics must contain a tx power rating" << endl;
    exit (EXIT_FAILURE);
  }
  if (!config["tx"].is_number ()) {
    cerr << "tx power rating must be a number" << endl;
    exit (EXIT_FAILURE);
  }
  if (!config.contains ("rx")) {
    cerr << "energy communication statistics must contain a rx power rating" << endl;
    exit (EXIT_FAILURE);
  }
  if (!config["rx"].is_number ()) {
    cerr << "rx power rating must be a number" << endl;
    exit (EXIT_FAILURE);
  }
  _config = EnergyCommunicationConfig (
    config["tx"].template get<double> (),
    config["rx"].template get<double> ()
  );
}

void EnergyCommunicationStatistics::add (BasicEventHandlerData *ehd) {
  this->addInternal ((SendPacketEventHandlerData *) ehd);
}

void EnergyCommunicationStatistics::addInternal (SendPacketEventHandlerData *sehd) {
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
  }
  this->_blocks.emplace_back (start, end, tsize, psize, asize, rsize);
}

void EnergyCommunicationStatistics::process (void) {
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

void EnergyCommunicationStatistics::report (void) {
  cout << "Total Transmit Energy Consumed: " << (double) this->_totalTSize * 8.0 * this->_config._pJPerBitTx * 1.0e-12 << " J" << endl;
  cout << "Total Receive Energy Consumed: " << (double) this->_totalRSize * 8.0 * this->_config._pJPerBitRx * 1.0e-12 << " J" << endl;
}

const vector<StatisticsType> &EnergyCommunicationStatistics::targets (void) const {
  return this->_targets;
}
