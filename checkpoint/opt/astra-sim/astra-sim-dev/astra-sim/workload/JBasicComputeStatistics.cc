#include "astra-sim/workload/JBasicComputeStatistics.hh"

#include <iostream>

using namespace Jalil;
using namespace AstraSim;
using namespace std;

ComputeBlock::ComputeBlock (
    uint64_t node, uint64_t start, uint64_t end, double cu, double mu, double power) :
  _node (node), _start (start), _end (end), _cu (cu), _mu (mu), _power (power) {}

const vector<StatisticsType> BasicComputeStatistics::_targets = {
  StatisticsType::Computation  
};

BasicComputeStatistics::BasicComputeStatistics (void)
  : StatisticsProcessor (), _aggCompUtilization (0), _aggMemUtilization (0), _time (0) {}

void BasicComputeStatistics::add (AstraSim::BasicEventHandlerData *ehd) {
  this->addInternal ((Jalil::ComputeEventHandlerData *) ehd);
}

void BasicComputeStatistics::addInternal (Jalil::ComputeEventHandlerData *cehd) {
  this->_blocks.emplace_back (
    cehd->sys_id,
    cehd->start_time,
    cehd->end_time,
    cehd->compute_utilization,
    cehd->memory_utilization,
    cehd->power
  );
}

void BasicComputeStatistics::process (void) {
  for (auto &block : _blocks) {
    double time = block._end - block._start;
    this->_aggCompUtilization = this->_aggCompUtilization + block._cu * time;
    this->_aggMemUtilization = this->_aggMemUtilization + block._cu * time;
    this->_time = block._end;
  }
  this->_blocks.clear ();
}

void BasicComputeStatistics::report (void) {
  cout << "Compute Utilization: " << this->_aggCompUtilization / this->_time << endl;
  cout << "Memory Utilization: " << this->_aggMemUtilization / this->_time << endl;
}

const vector<StatisticsType> &BasicComputeStatistics::targets (void) const {
  return _targets;
}
