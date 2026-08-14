#ifndef __J_ENERGY_COMMUNICATION_STATISTICS_HH__
#define __J_ENERGY_COMMUNICATION_STATISTICS_HH__

#include "astra-sim/system/BasicEventHandlerData.hh"
#include "astra-sim/system/JPacketTracker.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "astra-sim/workload/JStatistics.hh"
#include <json/json.hpp>

#include <array>
#include <vector>

namespace Jalil {

class EnergyCommunicationConfig {
  public:
    EnergyCommunicationConfig (
      std::vector<int> nodes,
      double pJPerBitTx,
      double pJPerBitRx
    );

    std::vector<int> _nodes;
    double _pJPerBitTx;
    double _pJPerBitRx;
};

class EnergyCommunicationStatistics : public StatisticsProcessor {
  public:
    EnergyCommunicationStatistics (nlohmann::json config);
    void add (AstraSim::BasicEventHandlerData *ehd) override;
    void process (void) override;
    void report (void) override;
    const std::vector<StatisticsType> &targets (void) const override;
  private:
    void addInternal (AstraSim::SendPacketEventHandlerData *sehd);

    std::vector<EnergyCommunicationConfig> _configs;
    std::vector<uint64_t> _totalTxSize;
    std::vector<uint64_t> _totalRxSize;
    static const std::vector<StatisticsType> _targets;
};


} // namespace Jalil

#endif /* __J_BASIC_COMMUNICATION_STATISTICS_HH__ */
