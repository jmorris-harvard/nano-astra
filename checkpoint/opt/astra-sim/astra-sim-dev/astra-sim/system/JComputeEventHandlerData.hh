#ifndef __COMPUTE_EVENT_HANDLER_DATA_HH__
#define __COMPUTE_EVENT_HANDLER_DATA_HH__

#include "astra-sim/system/BasicEventHandlerData.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"

namespace Jalil {

class ComputeEventHandlerData : public AstraSim::BasicEventHandlerData {
  public:
    ComputeEventHandlerData ();
    AstraSim::WorkloadLayerHandlerData *wlhd;
    // statistics
    double compute_utilization;
    double memory_utilization;
};

} // namespace Jalil

#endif /* __COMPUTE_EVENT_HANDLER_DATA_HH__ */
