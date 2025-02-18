#!/usr/bin/python3

import argparse
import numpy as np
import os
import pandas as pd
import random
import sys
import yaml

from enum import Enum

from topgen import (
  A100,
  DGXA100,
  SUA100,
  SuperPodA100,
  SuperSpineA100,
  FatTree,
  LeafSpine,
  JellyFish
)

def generate (topology, filename, target_design, verbose = False, k = 2, N = 40, racks = 4, interconnects = 2):
  # determine which topology to load
  dut = None
  if topology.lower () == 'A100'.lower ():
    dut = A100 ()
  elif topology.lower () == 'DGXA100'.lower ():
    dut = DGXA100 ()
  elif topology.lower () == 'SUA100'.lower ():
    dut = SUA100 ()
  elif topology.lower () == 'SuperPodA100'.lower ():
    dut = SuperPodA100 ()
  elif topology.lower () == 'SuperSpineA100'.lower ():
    dut = SuperSpineA100 ()
  elif topology.lower () == 'FatTree'.lower ():
    dut = FatTree (k, N)
  elif topology.lower () == 'LeafSpine'.lower ():
    dut = LeafSpine (k, N)
  elif topology.lower () == 'JellyFish'.lower ():
    dut = JellyFish (k = k, P = N, N = racks, r = interconnects)
  else:
    print ('Invalid sample topology request given (%s)' % (topology))
    sys.exit (1)
  output = dut.flatten ()
  print ('compute nodes (%s)' % (len (output['comp'])))
  if verbose:
    for node in output['comp']:
      print (node)
    print ()
  print ('switches nodes (%s)' % (len (output['switch'])))
  if verbose:
    for node in output['switch']:
      print (node)
    print ()
  cl = [link for link in output['link'] if link.n () == 2]
  print ('link nodes (%s)' % (len (cl)))
  if verbose:
    for node in output['link']:
      print (node)
  # read in configuratation
  yml = None
  try:
    with open (filename, 'r') as file:
      yml = yaml.load (file, Loader = Loader)
  except:
    yml = {}
  map = {}
  i = None
  for i, node in enumerate (output['comp']):
    map[node._id] = i
  # increment because enumerate does no overshoot
  i = i + 1
  for j, switch in enumerate (output['switch']):
    map[switch._id] = i + j
  if target_design not in yml.keys ():
    # add design
    yml[target_design] = {}
    design = yml[target_design]
    design['type'] = 'ns3:dev'
    design['ns3'] = {}
    design['ns3']['topology'] = {}
    # add dims
    design['dims'] = []
    design['dims'].append ({})
    design['dims'][0]['valid'] = True
    # also add collective implementations (required) by configure-astra
    # set all to halving doubling doesnt matter
    design['dims'][0]['all-reduce'] = 'halving-doubling'
    design['dims'][0]['all-gather'] = 'halving-doubling'
    design['dims'][0]['all-to-all'] = 'halving-doubling'
    design['dims'][0]['reduce-scatter'] = 'halving-doubling'
    # add empty workload section to be populated later
    design['workloads'] = [] 
    # add power
    design['power'] = {}
    design['power']['link'] = []
    design['power']['compute'] = []

  # set up compute
  design['dims'][0]['npus'] = len (output['comp'])
  # only considers 1st node for power and throughput
  design['dims'][0]['peak-perf'] = output['comp'][0].perf () * 1e-12
  design['dims'][0]['local-mem-bw'] = output['comp'][0].bw () * 1e-9
  design['power']['compute'].append ({
    'nodes': [],
    'idle': output['comp'][0].idle (),
    'peak': output['comp'][0].peak ()
  })

  # set 
  design['ns3']['topology']['switches'] = len (output['switch'])
  design['ns3']['topology']['links'] = []
  linkpowers = {}
  for j, link in enumerate (output['link']):
    if link.bnode () is not None:
      l = {}
      l['node-a'] = map[link.anode ().id ()]
      l['node-b'] = map[link.bnode ().id ()]
      # put bandwidth in Gbps
      l['bandwidth'] = '%.1fGbps' % (link.bw () * 1e-9)
      # put in ms
      l['latency'] = '%.6fms' % (link.latency () * 1e3)
      # change later
      l['error-rate'] = 0.0
      design['ns3']['topology']['links'].append (l)
      # add power
      if type (link) not in linkpowers:
        lg = {
          'links': [],
          'idle': link.idle (),
          'peak': link.peak ()
        }
        linkpowers[type (link)] = len (design['power']['link'])
        design['power']['link'].append (lg)
      # append to node list
      linkdesc = {
        'node-a': l['node-a'],
        'node-b': l['node-b']
      }
      design['power']['link'][linkpowers[type (link)]]['links'].append (linkdesc)

  # dump out contents
  with open (filename, 'w') as file:
    yaml.dump (yml, file, default_flow_style = False)

def main ():
  parser = argparse.ArgumentParser (
    prog = 'ASTRA-topology',
    description = '',
    epilog = '')
  parser.add_argument ('-c','--config', default='configuration.yaml')
  parser.add_argument ('-d','--design', required=True)
  parser.add_argument ('-t','--topology', required=True)
  parser.add_argument ('-k','--pods', default=2)
  parser.add_argument ('-N','--ports', default=40)
  parser.add_argument ('-R','--racks', default=4)
  parser.add_argument ('-r','--interconnects', default=2)
  parser.add_argument ('-v','--verbose', action='store_true', default = False)
  args = parser.parse_args (sys.argv[1:])
  generate (args.topology, args.config, args.design, args.verbose, int (args.pods), int (args.ports), int (args.racks), int (args.interconnects))

if __name__ == '__main__':
  main ()
