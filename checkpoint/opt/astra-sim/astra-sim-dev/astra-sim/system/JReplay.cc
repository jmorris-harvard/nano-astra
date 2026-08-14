#include "astra-sim/system/JCustom.hh"
#include "astra-sim/system/JReplay.hh"

using namespace std;
using namespace AstraSim;
using namespace Jalil;

using json = nlohmann::json;

CustomReplay::CustomReplay (Sys *sys, json config)
    : CustomCompute (sys, CustomComputeType::REPLAY) {}

double CustomReplay::runtime (shared_ptr<Chakra::FeederV3::ETFeederNode> node, ComputeEventHandlerData *cehd) {
  cehd->compute_utilization = 0.0;
  cehd->memory_utilization = 0.0;
  cehd->power = node->get_attr<double> ("live_power", 0.0);
  double live_runtime = node->get_attr<double> ("live_runtime", 0.0); // ns
  double elapsed_time = live_runtime * 1.0e-9; // ns -> sec
  return elapsed_time;
}
