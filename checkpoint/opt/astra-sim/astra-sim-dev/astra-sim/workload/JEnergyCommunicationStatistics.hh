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

class EnergyCommunicationBlock {
  public:
    EnergyCommunicationBlock (
      uint64_t start,
      uint64_t end,
      uint64_t tsize,
      uint64_t psize,
      uint64_t asize,
      uint64_t rsize
    );

    uint64_t _start;
    uint64_t _end;
    uint64_t _tsize;
    uint64_t _psize;
    uint64_t _asize;
    uint64_t _rsize;
};

class EnergyCommunicationConfig {
  public:
    EnergyCommunicationConfig (
      double pJPerBitTx,
      double pJPerBitRx
    );

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

    std::vector<EnergyCommunicationBlock> _blocks;
    EnergyCommunicationConfig _config;
    uint64_t _aggTotalBytesSent;
    uint64_t _aggTransmitTime;
    uint64_t _totalTransmitEvents;
    uint64_t _totalTSize;
    uint64_t _totalPSize;
    uint64_t _totalASize;
    uint64_t _totalRSize;
    static const std::vector<StatisticsType> _targets;
};


} // namespace Jalil

#endif /* __J_BASIC_COMMUNICATION_STATISTICS_HH__ */
