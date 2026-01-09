#include "astra-sim/workload/JEnergyComputeStatistics.hh"
#include "astra-sim/system/Sys.hh"

#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace Jalil;
using namespace AstraSim;
using namespace std;
using json = nlohmann::json;

EnergyComputeBlock::EnergyComputeBlock (
    uint64_t node, uint64_t start, uint64_t end, double cu, double mu) :
  _node (node), _start (start), _end (end), _cu (cu), _mu (mu) {}

EnergyConfig::EnergyConfig (
    vector<int> nodes, double idle, double cPeak, double mPeak, double tPeak) :
  _nodes (std::move (nodes)), _idle (idle), _cPeak (cPeak), _mPeak (mPeak), _tPeak (tPeak) {}

const vector<StatisticsType> EnergyComputeStatistics::_targets = {
  StatisticsType::Computation
};

EnergyComputeStatistics::EnergyComputeStatistics (json config)
  : StatisticsProcessor (), _computeEnergyConsumed (0.0), _memoryEnergyConsumed (0.0) {
  if (!config.contains ("rooflines")) {
    cerr << "energy compute statistics must contain roofline configs" << endl;
    exit (EXIT_FAILURE);
  }
  if (!config["rooflines"].is_array ()) {
    cerr << "roofline configs must be string" << endl;
    exit (EXIT_FAILURE);
  }
  auto rooflines = config["rooflines"].template get<vector<json>> ();
  for (json &roofline : rooflines) {
    if (!roofline.contains ("nodes")) {
      cerr << "each roofline must contain assigned nodes" << endl;
      exit (EXIT_FAILURE);
    }
    if (!roofline["nodes"].is_array ()) {
      cerr << "assigned nodes must be array" << endl;
      exit (EXIT_FAILURE);
    }
    if (!roofline.contains ("idle")) {
      cerr << "each roofline must contain an idle power" << endl;
      exit (EXIT_FAILURE);
    }
    if (!roofline["idle"].is_number ()) {
      cerr << "idle power must be number" << endl;
      exit (EXIT_FAILURE);
    }
    if (!roofline.contains ("compute_power")) {
      cerr << "each roofline must contain a peak compute power" << endl;
      exit (EXIT_FAILURE);
    }
    if (!roofline["compute_power"].is_number ()) {
      cerr << "peak compute power must be number" << endl;
      exit (EXIT_FAILURE);
    }
    if (!roofline.contains ("memory_power")) {
      cerr << "each roofline must contain a peak memory power" << endl;
      exit (EXIT_FAILURE);
    }
    if (!roofline["memory_power"].is_number ()) {
      cerr << "peak memory power must be number" << endl;
      exit (EXIT_FAILURE);
    }
    if (!roofline.contains ("tdp")) {
      cerr << "each roofline must contain a tdp" << endl;
      exit (EXIT_FAILURE);
    }
    if (!roofline["tdp"].is_number ()) {
      cerr << "tdp must be number" << endl;
      exit (EXIT_FAILURE);
    }
    _configs.emplace_back (
      roofline["nodes"].template get<vector<int>> (),
      roofline["idle"].template get<double> (),
      roofline["compute_power"].template get<double> (),
      roofline["memory_power"].template get<double> () * 1e-12, // in pJ/B
      roofline["tdp"].template get<double> ()
    );
  }
}

void EnergyComputeStatistics::add (AstraSim::BasicEventHandlerData *ehd) {
  this->addInternal ((Jalil::ComputeEventHandlerData *) ehd);
}

void EnergyComputeStatistics::addInternal (Jalil::ComputeEventHandlerData *cehd) {
  this->_blocks.emplace_back (
    cehd->sys_id,
    cehd->start_time,
    cehd->end_time,
    cehd->compute_utilization,
    cehd->memory_utilization
  );
}

void EnergyComputeStatistics::process (void) {
  for (auto &block : this->_blocks) {
    EnergyConfig *config = nullptr;
    for (auto &c : this->_configs) {
      if (std::find (c._nodes.begin (), c._nodes.end (), block._node) != c._nodes.end ()) {
        config = &c;
        break;
      }
    }
    double power = block._cu * config->_cPeak;
    this->_computeEnergyConsumed = this->_computeEnergyConsumed + ((double) block._end - (double) block._start) * 1.0e-9 * power;
    this->_memoryEnergyConsumed = this->_memoryEnergyConsumed + block._mu * config->_mPeak;
  }
  this->_blocks.clear ();
}

void EnergyComputeStatistics::report (void) {
  double idleEnergy = (double) Sys::boostedTick () * 1.0e-9 * this->_configs[0]._idle;
  cout << "Compute Energy Consumed: " << this->_computeEnergyConsumed + idleEnergy << " J" << endl;
  cout << "Memory Energy Consumed: " << this->_memoryEnergyConsumed << " J" << endl;
}

const vector<StatisticsType> &EnergyComputeStatistics::targets (void) const {
  return this->_targets;
}
