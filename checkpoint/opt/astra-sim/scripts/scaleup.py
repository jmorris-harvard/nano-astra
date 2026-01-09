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
  Node,
  GenericCompute,
  GenericEndpoint,
  PCIe4,
  PCIe5,
  NetworkCardSwitch,
  Infiniband,
  NVLink3,
  NVLinkSwitch,
  NVLink4,
  NVLinkSwitch3
)

def generate (dut, filename, target_design, verbose = False):
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
    yml[target_design] = dict ()
    design = yml[target_design]
    design['type'] = 'ns3:dev'
    design['ns3'] = dict ()
    design['ns3']['topology'] = dict ()
    # add dims
    design['dims'] = list ()
    design['dims'].append (dict ())
    design['dims'][0]['valid'] = True
    # set all to halving doubling
    design['dims'][0]['all-reduce'] = 'halving-doubling'
    design['dims'][0]['all-gather'] = 'halving-doubling'
    design['dims'][0]['all-to-all'] = 'halving-doubling'
    design['dims'][0]['reduce-scatter'] = 'halving-doubling'
    # add empty workload section
    design['workloads'] = list () 
    # add power
    design['power'] = {}
    design['power']['link'] = []
    design['power']['compute'] = []

  # set up compute 
  design['dims'][0]['npus'] = len (output['comp'])
  design['dims'][0]['peak-perf'] = output['comp'][0].perf () * 1e-12
  design['dims'][0]['local-mem-bw'] = output['comp'][0].bw () * 1e-9
  design['power']['compute'].append ({
    'nodes': [],
    'idle': output['comp'][0].idle (),
    'peak': output['comp'][0].peak ()
  })

  # set up statistics 
  design['dims'][0]['statistics-enabled'] = 1
  design['dims'][0]['statistics-config'] = list ()
  cstats = dict ()
  cstats['name'] = 'energy_compute'
  cstats['rooflines'] = list ()

  # set up custom compute
  design['dims'][0]['roofline-enabled'] = 0
  design['dims'][0]['custom-enabled'] = 1
  design['dims'][0]['custom-compute'] = list ()
  compute_map = dict ()
  for i, comp in enumerate (output['comp']):
    if comp.sid () not in compute_map:
      custom = dict ()
      custom['nodes'] = list ()
      custom['type'] = 'roofline'
      custom['config'] = dict ()
      custom['config']['throughput'] = comp.perf () * 1e-12
      custom['config']['bandwidth'] = comp.bw () * 1e-9
      custom['config']['l2'] = comp.l2 () * 1e-6
      compute_map[comp.sid ()] = len (design['dims'][0]['custom-compute'])
      design['dims'][0]['custom-compute'].append (custom)
      roofline = dict ()
      roofline['nodes'] = list ()
      roofline['idle'] = comp.idle_power ()
      roofline['compute_power'] = comp.compute_power ()
      roofline['memory_power'] = comp.memory_power ()
      roofline['tdp'] = comp.tdp ()
      cstats['rooflines'].append (roofline)
    compute = compute_map[comp.sid ()]
    design['dims'][0]['custom-compute'][compute]['nodes'].append (i)
    cstats['rooflines'][compute]['nodes'].append (i)
  design['dims'][0]['statistics-config'].append (cstats)

  # set up links 
  lstats = dict ()
  lstats['name'] = 'energy_comm'
  lstats['tx'] = 0.0
  lstats['rx'] = 0.0
  design['ns3']['topology']['switches'] = len (output['switch'])
  design['ns3']['topology']['links'] = []
  linkpowers = {}
  kmax = [0]
  kmin = [0]
  pmax = [0]
  speed_list = list ()
  for j, link in enumerate (output['link']):
    if link.bnode () is not None:
      l = {}
      l['node-a'] = map[link.anode ().id ()]
      l['node-b'] = map[link.bnode ().id ()]
      # put bandwidth in Gbps
      l['bandwidth'] = '%.1fGbps' % (link.bw () * 1e-9)
      if int (link.bw ()) not in speed_list:
        kmax.append (int (link.bw ()))
        kmax.append (800)
        kmax[0] = kmax[0] + 1
        kmin.append (int (link.bw ()))
        kmin.append (200)
        kmin[0] = kmin[0] + 1
        pmax.append (int (link.bw ()))
        pmax.append (0.2)
        pmax[0] = pmax[0] + 1
        speed_list.append (int (link.bw ()))
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
        lstats['tx'] = link.tx ()
        lstats['rx'] = link.rx ()
      # append to node list
      linkdesc = {
        'node-a': l['node-a'],
        'node-b': l['node-b']
      }
      design['power']['link'][linkpowers[type (link)]]['links'].append (linkdesc)
  design['dims'][0]['statistics-config'].append (lstats)
  design['ns3']['kmax-map'] = kmax
  design['ns3']['kmin-map'] = kmin
  design['ns3']['pmax-map'] = pmax

  # dump out contents
  with open (filename, 'w') as file:
    yaml.dump (yml, file, default_flow_style = False)

class A100Template (GenericCompute):
  def __init__ (this, Interconnect):
    super ().__init__ (13, [
      Interconnect, Interconnect, Interconnect,
      Interconnect, Interconnect, Interconnect,
      Interconnect, Interconnect, Interconnect,
      Interconnect, Interconnect, Interconnect,
      PCIe4
    ])
    this.bw (1250.0e9) # approximated effective bandwidth
    this.perf (280.0e12) # measured peak fp16 performance
    this.l2 (40e6) # standard l2 cache size
    this.sid ('A100SXM')
    this.idle_power (80.0) # measured idle power (when on)
    this.compute_power (420.0) # idle minus peak in small matmul
    this.memory_power (315.0) # based on load sweep benchmark (pJ/B)
    this.tdp (500.0) # standard tdp

class H200Template (GenericCompute):
  def __init__ (this, Interconnect):
    super ().__init__ (19, [
      Interconnect, Interconnect, Interconnect,
      Interconnect, Interconnect, Interconnect,
      Interconnect, Interconnect, Interconnect,
      Interconnect, Interconnect, Interconnect,
      Interconnect, Interconnect, Interconnect,
      Interconnect, Interconnect, Interconnect,
      PCIe5
    ])
    this.bw (4000.0e9) # approximated effective bandwidth
    this.perf (700.0e12) # measured peak performance
    this.l2 (50e6) # standard l2 cache size
    this.sid ('H200SXM')
    this.idle_power (115.0) # measured idle power (when on)
    this.compute_power (585.0) # idle minus peak in small matmul
    this.memory_power (121.0) # based on load sweep benchmark (pJ/B)
    this.tdp (700.0) # standard tdp

class A100Cluster (GenericEndpoint):
  def __init__ (this, n, Interconnect, Switch):
    super ().__init__ (0)
    this._type = Node.Type.CONTAINER
    this._nodes = [A100Template (Interconnect) for _ in range (n)] 
    if n in [1]:
      pass
    elif n in [2, 4]:
      multi = int (12 / (n - 1))
      for i, a100s in enumerate (this._nodes):
        for j, a100d in enumerate (this._nodes[i+1:]):
          for _ in range (multi):
            print ('link made')
            a100s.link (a100d, Interconnect)
    elif n in [8]:
      multi = 2
      this._int_switches = [Switch () for _ in range (6)]
      for a100 in this._nodes:
        for switch in this._int_switches:
          for _ in range (multi):
            a100.link (switch, Interconnect)
    else:
      print ('undefined cluster size:', n)
      sys.exit (1)
    this._out_switches = [NetworkCardSwitch (PCIe4, n, Infiniband, n)]
    for i, a100 in enumerate (this._nodes):
      a100.link (this._out_switches[0], PCIe4)
    for link in this._out_switches[0].ports ():
      if link.n () == 1:
        this.add (link)
    this._sid = 'A100Cluster'
 
class H200Cluster (GenericEndpoint):
  def __init__ (this, n, Interconnect, Switch):
    super ().__init__ (0)
    this._type = Node.Type.CONTAINER
    this._nodes = [H200Template (Interconnect) for _ in range (n)] 
    if n in [1]:
      pass
    elif n in [2, 4]:
      multi = int (18 / (n - 1))
      for i, h200s in enumerate (this._nodes):
        for j, h200d in enumerate (this._nodes[i+1:]):
          for _ in range (multi):
            h200s.link (h200d, Interconnect)
    elif n in [8]:
      multi = 4
      this._int_switches = [Switch () for _ in range (4)]
      for h200 in this._nodes:
        for switch in this._int_switches:
          for _ in range (multi):
            h200.link (switch, Interconnect)
    else:
      print ('undefined cluster size:', n)
      sys.exit (1)
    this._out_switches = [NetworkCardSwitch (PCIe5, n, Infiniband, n)]
    for i, h200 in enumerate (this._nodes):
      h200.link (this._out_switches[0], PCIe5)
    for link in this._out_switches[0].ports ():
      if link.n () == 1:
        this.add (link)
    this._sid = 'H200Cluster'
  
def stringToClass (descriptor):
  if descriptor.lower () == 'a100':
    return A100Cluster
  elif descriptor.lower () == 'h200':
    return H200Cluster
  elif descriptor.lower () == 'nvlink3':
    return NVLink3
  elif descriptor.lower () == 'nvlink4':
    return NVLink4
  else:
    print ('class descriptor unknown:', descriptor)
    sys.exit ()

def linkToSwitch (descriptor):
  if descriptor.lower () == 'nvlink3':
    return NVLinkSwitch
  elif descriptor.lower () == 'nvlink4':
    return NVLinkSwitch3
  else:
    print ('class descriptor unknown:', descriptor)
    sys.exit ()

def parse_arguments ():
  parser = argparse.ArgumentParser (
    prog = 'ASTRA-ISCA-2026',
    description = '',
    epilog = '')
  parser.add_argument ('--n', required = True)
  parser.add_argument ('--gpu', required = True)
  parser.add_argument ('--link', required = True)
  parser.add_argument ('--design', required=True)
  parser.add_argument ('--config', default='configuration.yaml')
  parser.add_argument ('--verbose', action='store_true', default = False)
  return parser.parse_args (sys.argv[1:])

def main ():
  args = parse_arguments ()
  GPU = stringToClass (args.gpu)
  Interconnect = stringToClass (args.link)
  Switch = linkToSwitch (args.link)
  DUT = GPU (int (args.n), Interconnect, Switch) 
  generate (DUT, args.config, args.design, args.verbose)

if __name__ == '__main__':
  main ()
