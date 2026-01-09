#include "astra-sim/workload/JStatistics.hh"
#include "astra-sim/workload/JBasicComputeStatistics.hh"
#include "astra-sim/workload/JEnergyComputeStatistics.hh"
#include "astra-sim/workload/JBasicCommunicationStatistics.hh"
#include "astra-sim/workload/JEnergyCommunicationStatistics.hh"

#include <iostream>
#include <climits>
#include <string>

using namespace std;
using namespace AstraSim;
using namespace Jalil;
using json = nlohmann::json;

StatisticsProcessor::StatisticsProcessor (void) {}

StatisticsProcessor *StatisticsProcessorBuilder::build (json config) {
  if (!config.contains ("name")) {
    exit (EXIT_FAILURE);
  }
  if (!config["name"].is_string ()) {
    exit (EXIT_FAILURE);
  }

  auto type = config["name"].template get<string> ();
  transform (type.begin(), type.end(), type.begin(), [](unsigned char c){ return std::tolower(c); } );
  if (type.compare ("basic_compute") == 0) {
    // std::cout << "building compute processor" << std::endl;
    return new BasicComputeStatistics ();
  } else if (type.compare ("basic_comm") == 0) {
    // std::cout << "building comm processor" << std::endl;
    return new BasicCommunicationStatistics ();
  } else if (type.compare ("energy_compute") == 0) {
    // std::cout << "building energy compute processor" << std::endl;
    return new EnergyComputeStatistics (config);
  } else if (type.compare ("energy_comm") == 0) {
    // std::cout << "building energy communication processor" << std::endl;
    return new EnergyCommunicationStatistics (config);
  } else {
    std::cerr << "processor type unknown" << std::endl;
    exit (EXIT_FAILURE);
  }
}

Statistics::Statistics (json config) {
  if (!config.is_array ()) {
    std::cerr << "config must be an array" << std::endl;
    exit (EXIT_FAILURE);
  }

  // std::cout << "building processors" << std::endl;
  auto processors = config.template get<vector<json>> ();
  for (auto processor : processors) {
    this->_processors.push_back (StatisticsProcessorBuilder::build (processor));
  }
  // std::cout << "built processors" << std::endl;
}

Statistics::~Statistics () {
  /*
  for (auto processor : this->_processors) {
    delete processor;
  }
  */
}

void Statistics::add (StatisticsType target, BasicEventHandlerData *ehd) {
  for (auto processor : this->_processors) {
    for (auto t : processor->targets ()) {
      if (target == t) {
        processor->add (ehd);
      }
    }
  }
}

void Statistics::process (void) {
  for (auto processor : this->_processors) {
    processor->process ();
  }
}

void Statistics::report (void) {
  for (auto processor : this->_processors) {
    processor->report ();
  }
}
