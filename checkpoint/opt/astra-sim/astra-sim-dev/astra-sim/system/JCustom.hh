#ifndef __CUSTOMCOMPUTE_HH__
#define __CUSTOMCOMPUTE_HH__

#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/JComputeEventHandlerData.hh"
#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"
#include <json/json.hpp>

#include <memory>

namespace Jalil {

enum class CustomComputeType {
  INVALID = 0,
  ROOFLINE,
  CUDA
};

class CustomCompute {
  public:
    CustomCompute (AstraSim::Sys *sys, CustomComputeType type);
    CustomComputeType type (void);
    virtual double runtime (std::shared_ptr<Chakra::FeederV3::ETFeederNode> node, ComputeEventHandlerData * cehd) = 0;
    virtual ~CustomCompute (void) {}
  protected:
    AstraSim::Sys *_sys;
  private:
    CustomComputeType _type;
};

class CustomComputeBuilder {
  public:
    CustomComputeBuilder (void) = delete;
    static CustomCompute *build (AstraSim::Sys *sys, nlohmann::json config);
};

} // namespace Jalil

#endif /* __CUSTOMCOMPUTE_HH__ */
