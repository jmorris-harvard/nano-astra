#!/usr/bin/python3

import argparse
import copy
import glob
import importlib
import os
import re
import shutil
import subprocess
import sys
import yaml

import etgenerate
from astragen import encode, GlobalMetadata
try:
  from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
  from yaml import Loader, Dumper

class Analytical ():
  NETWORK = {
    # -- required arguments
    'req': {
    'topology': [],
      'npus': [],
      'bandwidth': [],
      'latency': []
    },
    # -- optional arguments
    'opt': {}
  }

  SYSTEM = {
    # -- required arguments
    'req': {
      'all-reduce': [],
      'all-gather': [],
      'reduce-scatter': [],
      'all-to-all': []
    },
    # -- optional arguments
    'opt': {
      'scheduling-policy': 'LIFO',
      'roofline-enabled': 0,
      'replay-only': 0,
      'endpoint-delay': 0,
      'preferred-dataset-splits': 4,
      'active-chunks-per-dimension': 1,
      'collective-optimization': 'localBWAware',
      'local-mem-bw': 50,
      'boost-mode': 0,
      'peak-perf': 0.001, # Measured in TFLOPS
      'custom-enabled': 0,
      'custom-compute': []
    }
  }

class NS3 ():
  NETWORK = {
    # -- required arguments
    'req': {
      'npus': [] 
    },
    # -- optional arguments
    'opt': {}
  }

  SYSTEM = {
    # -- required arguments
    'req': {
      'all-reduce': [],
      'all-gather': [],
      'reduce-scatter': [],
      'all-to-all': []
    },
    # -- optional arguments
    'opt': {
      'scheduling-policy': 'LIFO',
      'roofline-enabled': 0,
      'replay-only': 0,
      'endpoint-delay': 0,
      'preferred-dataset-splits': 4,
      'active-chunks-per-dimension': 1,
      'collective-optimization': 'localBWAware',
      'local-mem-bw': 50,
      'boost-mode': 0,
      'peak-perf': 0.001, # Measured in TFLOPS
      'custom-enabled': 0,
      'custom-compute': []
    }
  }

  NS3ARGS = {
    # -- required arguments
    'req': {
      'topology': [],
      'topology-file': None
    },
    # -- optional arguments
    'opt': { 
      'enable-qcn': 1,
      'use-dynamic-pfc-threshold': 1,
      'packet-payload-size': 1024,
      'flow-file': None,
      'trace-file': None,
      'trace-output-file': None,
      'fct-output-file': None,
      'pfc-output-file': None,
      'qlen-mon-file': None,
      'qlen-mon-start': 0,
      'qlen-mon-end': 20000,
      'simulator-stop-time': 4000000.0,
      'cc-mode': 12,
      'alpha-resume-interval': 1,
      'rate-decrease-interval': 1,
      'clamp-target-rate': 0,
      'rp-timer': 900,
      'ewma-gain': 0.00390625,
      'fast-recovery-times': 1,
      'rate-ai': '50Mb/s',
      'rate-hai': '100Mb/s',
      'min-rate': '100Mb/s',
      'dctcp-rate-ai': '1000Mb/s',
      'error-rate-per-link': 0.00,
      'l2-chunk-size': 4000,
      'l2-ack-interval': 1,
      'l2-back-to-zero': 0,
      'has-win': 1,
      'global-t': 0,
      'var-win': 1,
      'fast-react': 1,
      'u-target': 0.95,
      'mi-thresh': 0,
      'int-multi': 1,
      'multi-rate': 0,
      'sample-feedback': 0,
      'pint-log-base': 1.05,
      'pint-prob': 1.0,
      'nic-total-pause-time': 0,
      'rate-bound': 1,
      'ack-high_prio': 0,
      'link-down': [0, 0, 0],
      'enable-trace': 1,
      'kmax-map': [
          (25000000000, 400), 
          (40000000000, 800),
          (200000000000, 1600),
          (400000000000, 2400),
          (800000000000, 3200),
          (1600000000000, 3200)
      ],
      'kmin-map': [
          (25000000000, 100),
          (40000000000, 200),
          (200000000000, 400),
          (400000000000, 600),
          (800000000000, 800),
          (1600000000000, 800)
      ],
      'pmax-map': [
          (25000000000, 0.2),
          (40000000000, 0.2),
          (200000000000, 0.2),
          (400000000000, 0.2),
          (800000000000, 0.2),
          (1600000000000, 0.2)
      ],
      'buffer-size': 32
    }
  }


def generate (filename, target, workload, overwrite = True):
  # read configuration
  yml = None
  with open (filename, 'r') as file:
    yml = yaml.load (file, Loader = Loader)
  if target_design not in yml.keys ():
    print ('Could not find (%s) in (%s)' % (target_design, filename))
  design = yml[target_design]

  # select flavor
  flavor, tag = design['type'].split (':')
  network = None
  system = None
  ns3 = None
  binary = 'astra-sim'
  if flavor == 'analytical':
    network = copy.deepcopy (Analytical.NETWORK)
    system = copy.deepcopy (Analytical.SYSTEM)
  elif flavor == 'ns3':
    network = copy.deepcopy (NS3.NETWORK)
    system = copy.deepcopy (NS3.SYSTEM)
    ns3 = copy.deepcopy (NS3.NS3ARGS)
    binary = binary + '-ns3'
  else:
    print ('Invalid type give (%s)' % flavor)
    sys.exit (1)
  if tag == 'base':
    pass
  elif tag == 'dev':
    binary = binary + '-dev'
  else:
    print ('Invalid tag give (%s)' % tag)
    sys.exit (1)
