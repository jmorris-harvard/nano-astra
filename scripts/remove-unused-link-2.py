#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys
import yaml

try:
  from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
  from yaml import Loader, Dumper

def usage (utilization, threshold):
  links = {}
  lines = None
  with open (utilization, 'r') as util:
    lines = util.read ().splitlines ()[1:]
  for line in lines:
    item, util = line.split (',')
    if '_' not in item:
      continue
    if float (util) < threshold:
      continue
    a, b = item.split ('_')
    a, b = min (int (a), int (b)), max (int (a), int (b))
    links[(a, b)] = True
  return links

def prune (config_filename, target_design, links, override = False):
  yml = None
  with open (config_filename, 'r') as config:
    yml = yaml.load (config, Loader = Loader)
  design = yml[target_design]

  pruned_links = []
  for link in design['ns3']['topology']['links']:
    a, b = link['node-a'], link['node-b']
    a, b = min (a, b), max (a, b)
    usage = links.get ((a, b), None)
    if usage is None:
      print ('removed', a, b)
      continue
    else:
      pruned_links.append (link)

  old = len (design['ns3']['topology']['links'])
  new = len (pruned_links)
  percent = float (new) / float (old)
  print ('%.3f link utilization' % (percent))

  # write out config with new links
  if percent < 1.0:
    yml[target_design]['ns3']['topology']['links'] = pruned_links
    new_config_filename = None
    if override:
      new_config_filename = config_filename
    else:
      new_config_filename = os.path.splitext (config_filename)[0] + '_new.yaml'
    with open (new_config_filename, 'w') as config:
      yaml.dump (yml, config, default_flow_style = False)
    print ('new configuration written to,', new_config_filename)
  else:
    print ('no new configuration made')

def main (): 
  parser = argparse.ArgumentParser(
    prog='ASTRA-power',
    description='',
    epilog='')
  parser.add_argument('-u', '--utilization', required = True)
  parser.add_argument('-c', '--configuration', required = True)
  parser.add_argument('-d', '--design', required = True)
  parser.add_argument('-t', '--threshold', required = True)
  parser.add_argument('-o', '--override', default = False, action = 'store_true')
  args = parser.parse_args (sys.argv[1:])
  # count link usage
  links = usage (args.utilization, float (args.threshold))
  # prune links
  prune (args.configuration, args.design, links, args.override)

if __name__ == '__main__':
  main ()
