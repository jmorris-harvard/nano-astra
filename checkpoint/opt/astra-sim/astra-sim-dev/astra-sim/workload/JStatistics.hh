#ifndef __J_STATISTICS_HH__
#define __J_STATISTICS_HH__

#include "astra-sim/common/Common.hh"
#include "astra-sim/system/BasicEventHandlerData.hh"
#include <json/json.hpp>

#include <memory>
#include <unordered_map>
#include <vector>


namespace Jalil {

enum class StatisticsType {
  Invalid = 0,
  Computation,
  Communication
};

class StatisticsProcessor {
  public:
    StatisticsProcessor (void);
    virtual void add (AstraSim::BasicEventHandlerData *ehd) = 0;
    virtual void process (void) = 0;
    virtual void report (void) = 0;
    virtual const std::vector<StatisticsType> &targets (void) const = 0;
};

class StatisticsProcessorBuilder {
  public:
    StatisticsProcessorBuilder (void) = delete;
    static StatisticsProcessor *build (nlohmann::json config);
};

class Statistics {
  public:
    Statistics (nlohmann::json config);

    void add (StatisticsType target, AstraSim::BasicEventHandlerData *ehd);
    void process (void);
    void report (void);

    ~Statistics (void);
  private:
    std::vector<StatisticsProcessor *> _processors;
};

} // namespace Jalil

#endif /* __J_STATISTICS_HH__ */
