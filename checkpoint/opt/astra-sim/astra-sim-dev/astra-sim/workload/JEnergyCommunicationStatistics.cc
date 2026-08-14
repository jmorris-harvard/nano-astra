#include "astra-sim/workload/JEnergyCommunicationStatistics.hh"

#include <algorithm>
#include <iostream>
#include <cstdlib>

using namespace Jalil;
using namespace AstraSim;
using namespace std;
using json = nlohmann::json;

EnergyCommunicationConfig::EnergyCommunicationConfig (vector<int> nodes, double pJPerBitTx, double pJPerBitRx)
  : _nodes (std::move (nodes)), _pJPerBitTx (pJPerBitTx), _pJPerBitRx (pJPerBitRx) {}

const vector<StatisticsType> EnergyCommunicationStatistics::_targets = {
  StatisticsType::Communication  
};

EnergyCommunicationStatistics::EnergyCommunicationStatistics (json config)
  : StatisticsProcessor () {
  if (!config.contains ("efficiencies")) {
    cerr << "energy communication statistics must contain efficiency configs" << endl;
    exit (EXIT_FAILURE);
  }
  if (!config["efficiencies"].is_array ()) {
    cerr << "efficiency configs must be array" << endl;
    exit (EXIT_FAILURE);
  }
  auto efficiencies = config["efficiencies"].template get<vector<json>> ();
  for (json &efficiency : efficiencies) {
    if (!efficiency.contains ("nodes")) {
      cerr << "each efficiency must contain assigned nodes" << endl;
      exit (EXIT_FAILURE);
    }
    if (!efficiency["nodes"].is_array ()) {
      cerr << "assigned nodes must be array" << endl;
      exit (EXIT_FAILURE);
    }
    if (!efficiency.contains ("tx")) {
      cerr << "each efficiency must contain a tx power rating" << endl;
      exit (EXIT_FAILURE);
    }
    if (!efficiency["tx"].is_number ()) {
      cerr << "tx power rating must be a number" << endl;
      exit (EXIT_FAILURE);
    }
    if (!efficiency.contains ("rx")) {
      cerr << "each efficiency must contain a rx power rating" << endl;
      exit (EXIT_FAILURE);
    }
    if (!efficiency["rx"].is_number ()) {
      cerr << "rx power rating must be a number" << endl;
      exit (EXIT_FAILURE);
    }
    _configs.emplace_back (
      efficiency["nodes"].template get<vector<int>> (),
      efficiency["tx"].template get<double> (),
      efficiency["rx"].template get<double> ()
    );
    this->_totalTxSize.push_back (0);
    this->_totalRxSize.push_back (0);
  }
}

void EnergyCommunicationStatistics::add (BasicEventHandlerData *ehd) {
  this->addInternal ((SendPacketEventHandlerData *) ehd);
}

void EnergyCommunicationStatistics::addInternal (SendPacketEventHandlerData *sehd) {
  const auto &packetShadows = sehd->_packetTracker->readPackets ();
  for (uint32_t i = 0; i < packetShadows.size (); ++i) {
    if (packetShadows[i]._loc != PacketLocation::TRANSMIT &&
        packetShadows[i]._loc != PacketLocation::RECEIVE) {
      continue;
    }
    int node = (int) packetShadows[i]._node;
    for (uint32_t c = 0; c < this->_configs.size (); ++c) {
      if (std::find (this->_configs[c]._nodes.begin (), this->_configs[c]._nodes.end (), node) ==
          this->_configs[c]._nodes.end ()) {
        continue;
      }
      if (packetShadows[i]._loc == PacketLocation::TRANSMIT) {
        this->_totalTxSize[c] = this->_totalTxSize[c] + packetShadows[i]._size * 8;
      } else {
        this->_totalRxSize[c] = this->_totalRxSize[c] + packetShadows[i]._size * 8;
      }
      break;
    }
  }
}

void EnergyCommunicationStatistics::process (void) {
}

void EnergyCommunicationStatistics::report (void) {
  for (uint32_t i = 0; i < this->_configs.size (); ++i) {
    double txEnergy = (double) this->_totalTxSize[i] * this->_configs[i]._pJPerBitTx * 1.0e-12;
    double rxEnergy = (double) this->_totalRxSize[i] * this->_configs[i]._pJPerBitRx * 1.0e-12;
    cout << "[report] Config " << i << " Transmit Energy Consumed: " << txEnergy << " J" << endl;
    cout << "[report] Config " << i << " Receive Energy Consumed: " << rxEnergy << " J" << endl;
  }
}

const vector<StatisticsType> &EnergyCommunicationStatistics::targets (void) const {
  return this->_targets;
}
