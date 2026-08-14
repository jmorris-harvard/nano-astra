#ifndef __J_ENERGY_COMPUTE_STATISTICS_HH__
#define __J_ENERGY_COMPUTE_STATISTICS_HH__

#include "astra-sim/system/BasicEventHandlerData.hh"
#include "astra-sim/system/JComputeEventHandlerData.hh"
#include "astra-sim/workload/JStatistics.hh"
#include <json/json.hpp>

#include <array>

namespace Jalil {

class EnergyComputeBlock {
  public:
    EnergyComputeBlock (
      uint64_t node,
      uint64_t start,
      uint64_t end,
      double cu,
      double mu,
      double power
    );

    uint64_t _node;
    uint64_t _start;
    uint64_t _end;
    double _cu;
    double _mu;
    double _power;
};

class EnergyConfig {
  public:
    EnergyConfig (
      std::vector<int> nodes,
      double idle,
      double cPeak,
      double mPeak,
      double tPeak
    );
    std::vector<int> _nodes;
    double _idle;
    double _cPeak;
    double _mPeak;
    double _tPeak;
};

class EnergyComputeStatistics : public StatisticsProcessor {
  public:
    EnergyComputeStatistics (nlohmann::json config);
    void add (AstraSim::BasicEventHandlerData *ehd) override;
    void process (void) override;
    void report (void) override;
    const std::vector<StatisticsType> &targets (void) const override;
  private:
    void addInternal (ComputeEventHandlerData *cehd);

    std::vector<EnergyComputeBlock> _blocks;
    std::vector<EnergyConfig> _configs;
    double _computeEnergyConsumed; 
    double _memoryEnergyConsumed; 
    static const std::vector<StatisticsType> _targets;
};


} // namesapce Jalil

#endif /* __J_BASIC_COMPUTE_STATISTICS_HH__ */
