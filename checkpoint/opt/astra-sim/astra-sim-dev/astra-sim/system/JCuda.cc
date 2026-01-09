#include "astra-sim/system/JCustom.hh"
#include "astra-sim/system/JCuda.hh"

#include <algorithm>
#include <cstdlib> 
#include <iostream>

using namespace std;
using namespace AstraSim;
using namespace Jalil;

using json = nlohmann::json;

CustomCUDA::CustomCUDA (Sys *sys, json config)
    : CustomCompute (sys, CustomComputeType::CUDA) {
  if (!config.contains ("sm")) {
    std::cout << "no sm count found" << std::endl;
    exit (EXIT_FAILURE);
  }
  if (!config["sm"].is_number ()) {
    std::cout << "sm count is not valid" << std::endl;
    exit (EXIT_FAILURE);
  }
  this->_totalSMs = config["sm"].template get<double> ();

  if (!config.contains ("memory")) {
    std::cout << "no memory data found" << std::endl;
    exit (EXIT_FAILURE);
  }
  if (!config["memory"].is_number ()) {
    std::cout << "memory is not valid" << std::endl;
    exit (EXIT_FAILURE);
  }
  this->_totalMemory = config["memory"].template get<double> () * 1e9;
}

double CustomCUDA::runtime (shared_ptr<Chakra::FeederV3::ETFeederNode> node, ComputeEventHandlerData *cehd) {
  // repurposing num ops to
  double sm = static_cast<double> (node->num_ops ());
  cehd->compute_utilization = sm / this->_totalSMs;
  double mem = static_cast<double> (node->tensor_size ());
  cehd->memory_utilization = mem / this->_totalMemory;
  double elapsed_time = 1.0e-9;
  if (node->runtime () != 0) {
    // assume runtime given in microseconds
    elapsed_time = static_cast<double> (node->runtime ()) * 1.0e-6;
  }
  return elapsed_time;
}
