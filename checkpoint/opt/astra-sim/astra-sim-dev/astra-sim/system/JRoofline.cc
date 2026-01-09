#include "astra-sim/system/JCustom.hh"
#include "astra-sim/system/JRoofline.hh"

#include <algorithm>
#include <iostream>

using namespace std;
using namespace AstraSim;
using namespace Jalil;

using json = nlohmann::json;

CustomRoofline::CustomRoofline (Sys *sys, json config) 
    : CustomCompute (sys, CustomComputeType::ROOFLINE) { 
  if (!config.contains ("throughput")) {
    std::cout << "no throughput found" << std::endl;
    exit (EXIT_FAILURE);
  }
  if (!config["throughput"].is_number ()) {
    std::cout << "throughput is not valid" << std::endl;
    exit (EXIT_FAILURE);
  }
  this->_throughput = config["throughput"].template get<double> () * 1e12; // in TFLOPS

  if (!config.contains ("bandwidth")) {
    std::cout << "no bandwidth found" << std::endl;
    exit (EXIT_FAILURE);
  }
  if (!config["bandwidth"].is_number ()) {
    std::cout << "bandwidth is not valid" << std::endl;
    exit (EXIT_FAILURE);
  }
  this->_bandwidth = config["bandwidth"].template get<double> () * 1e9; // in GBps

  if (!config.contains ("l2")) {
    std::cout << "no l2 cache size found" << std::endl;
    exit (EXIT_FAILURE);
  }
  if (!config["l2"].is_number ()) {
    std::cout << "l2 cache size is not valid" << std::endl;
    exit (EXIT_FAILURE);
  }
  this->_l2 = config["l2"].template get<double> () * 1e6; // in MB
}

double CustomRoofline::runtime (shared_ptr<Chakra::FeederV3::ETFeederNode> node, ComputeEventHandlerData *cehd) {
  double flops = static_cast<double> (node->num_ops ());
  double bytes = static_cast<double> (node->tensor_size ());
  double oi = flops / bytes;
  double effective = this->_bandwidth;
  double performance = min (effective * oi, this->_throughput);
  cehd->compute_utilization = performance / this->_throughput;
  cehd->memory_utilization = bytes;
  double elapsed_time = flops / performance;
  if (this->_sys->trace_enabled) {
  std::cout << "[tracker] " <<
		 "compute," << 
		 node->id () << "," <<
		 node->name () << "," <<
		 this->_sys->id << "," <<
		 Sys::boostedTick() << "," <<
		 node->num_ops () << "," <<
		 node->tensor_size () << "," <<
		 oi << "," <<
		 performance << "," <<
		 this->_throughput << "," <<
		 static_cast<uint64_t> (elapsed_time * 1.0e9) << std::endl;
  }
  return elapsed_time;
}
