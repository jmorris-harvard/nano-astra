#ifndef __CUSTOMCUDA_HH__
#define __CUSTOMCUDA_HH__

#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/JCustom.hh"
#include "astra-sim/system/JComputeEventHandlerData.hh"
#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"
#include <json/json.hpp>

#include <memory>

namespace Jalil {

class CustomCUDA : public CustomCompute {
 public:
  CustomCUDA (AstraSim::Sys *sys, nlohmann::json config);
  double runtime (std::shared_ptr<Chakra::FeederV3::ETFeederNode> node, ComputeEventHandlerData *cehd) override;
  ~CustomCUDA (void) override {}
 private:
  double _totalSMs;
  double _totalMemory;
};

} // namespace Jalil

#endif // __CUSTOMCUDA_HH__
