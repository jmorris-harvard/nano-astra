#ifndef __CUSTOMREPLAY_HH__
#define __CUSTOMREPLAY_HH__

#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/JCustom.hh"
#include "astra-sim/system/JComputeEventHandlerData.hh"
#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"
#include <json/json.hpp>

#include <memory>

namespace Jalil {

class CustomReplay : public CustomCompute {
  public:
    CustomReplay (AstraSim::Sys *sys, nlohmann::json config);
    double runtime (std::shared_ptr<Chakra::FeederV3::ETFeederNode> node, ComputeEventHandlerData *cehd) override;
    ~CustomReplay (void) override {}
};

} // namespace Jalil

#endif /* __CUSTOMREPLAY_HH__ */
