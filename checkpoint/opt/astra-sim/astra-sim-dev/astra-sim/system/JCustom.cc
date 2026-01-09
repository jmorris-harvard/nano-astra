#include "astra-sim/system/JCustom.hh"
#include "astra-sim/system/JRoofline.hh"
#include "astra-sim/system/JCuda.hh"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace Jalil;
using namespace AstraSim;
using json = nlohmann::json;

CustomCompute::CustomCompute (Sys *sys, CustomComputeType type) 
    : _sys (sys), _type (type) {}

CustomComputeType CustomCompute::type (void) {
  return _type;
}

CustomCompute *CustomComputeBuilder::build (Sys* sys, json config) {
  if (!config.is_array ()) {
    std::cout << "config must be an array" << std::endl;
    exit (EXIT_FAILURE);
  }

  auto vec = config.template get<vector<json>> ();
  bool found = false;
  json app;
  for ( auto &method : vec ) {
    if (!method.contains ("nodes")) {
      std::cout << "config must contain nodes element" << std::endl;
      exit (EXIT_FAILURE);
    }
    if (!method["nodes"].is_array ()) {
      std::cout << "nodes must be an array" << std::endl;
      exit (EXIT_FAILURE);
    }
    auto nodes = method["nodes"].template get<vector<int>> ();
    for ( auto &node : nodes ) {
      if (node == sys->id) {
      	found = true;
      }
    }
    if (found) {
      if (!method.contains ("config")) {
        std::cout << "config must contain config element" << std::endl;
        exit (EXIT_FAILURE);
      }
      app = method;
      break;
    }
  }

  if (!app.contains ("type")) {
    std::cout << "config must contain type" << std::endl;
    exit (EXIT_FAILURE);
  }
  if (!app["type"].is_string ()) {
    std::cout << "type must be a string" << std::endl;
    exit (EXIT_FAILURE);
  }

  // ADD CUSTOM COMPUTE TYPES HERE
  auto type = app["type"].template get<string> ();
  transform (type.begin(), type.end(), type.begin(), [](unsigned char c){ return std::tolower(c); } );
  if (type.compare ("roofline") == 0) {
    return new CustomRoofline (sys, app["config"]);
  } else if (type.compare ("cuda") == 0) {
    return new CustomCUDA (sys, app["config"]);
  } else {
    std::cout << "unknown type found" << std::endl;
    exit (EXIT_FAILURE);
  }
}
