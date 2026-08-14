#ifndef __J_BASIC_COMPUTE_STATISTICS_HH__
#define __J_BASIC_COMPUTE_STATISTICS_HH__

#include "astra-sim/system/BasicEventHandlerData.hh"
#include "astra-sim/system/JComputeEventHandlerData.hh"
#include "astra-sim/workload/JStatistics.hh"

#include <array>

namespace Jalil {

class ComputeBlock {
  public:
    ComputeBlock (
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

class BasicComputeStatistics : public StatisticsProcessor {
  public:
    BasicComputeStatistics (void);
    void add (AstraSim::BasicEventHandlerData *ehd) override;
    void process (void) override;
    void report (void) override;
    const std::vector<StatisticsType> &targets (void) const override;
  private:
    void addInternal (ComputeEventHandlerData *cehd);

    std::vector<ComputeBlock> _blocks;
    double _aggCompUtilization;
    double _aggMemUtilization;
    uint64_t _time;
    static const std::vector<StatisticsType> _targets;
};


} // namesapce Jalil

#endif /* __J_BASIC_COMPUTE_STATISTICS_HH__ */
