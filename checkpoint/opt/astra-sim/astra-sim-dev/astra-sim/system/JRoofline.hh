#ifndef __CUSTOMROOFLINE_HH__
#define __CUSTOMROOFLINE_HH__

#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/JCustom.hh"
#include "astra-sim/system/JComputeEventHandlerData.hh"
#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"
#include <json/json.hpp>

#include <memory>

namespace Jalil {

class CustomRoofline : public CustomCompute {
  public:
    CustomRoofline (AstraSim::Sys *sys, nlohmann::json config);
    double runtime (std::shared_ptr<Chakra::FeederV3::ETFeederNode> node, ComputeEventHandlerData *cehd) override;
    double throughput (void) { return _throughput; }
    double bandwidth (void) { return _bandwidth; }
    ~CustomRoofline (void) override {}
  private:
    double _throughput;
    double _bandwidth;
    double _l2;
};

} // namespace Jalil

#endif /* __CUSTOMROOFLINE_HH__ */
