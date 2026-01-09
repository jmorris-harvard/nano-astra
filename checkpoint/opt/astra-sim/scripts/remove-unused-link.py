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

def parse (topology_filename):
  # read design
  links = {
    'srcdst': {},
    'nodelink': {}
  }
  nodes = {}
  with open (topology_filename, 'r') as topology:
    lines = topology.read().splitlines ()[2:]
    pattern = r'(?P<nodea>[0-9]+)\s*' + \
              r'(?P<nodeb>[0-9]+)\s*' + \
              r'(?P<bandwidth>[0-9]+\.?([0-9]+)?\w*)\s*' + \
              r'(?P<latency>[0-9]+\.?([0-9]+)?\w*)\s*' + \
              r'(?P<errorrate>[0-9]+\.?([0-9]+)?\w*)\s*'
    for i, link in enumerate (lines):
      m = re.match (pattern, link)
      a = int (m.group ('nodea'))
      b = int (m.group ('nodeb'))
      a, b = min (a, b), max (a, b)
      # map (node, link) to (node, node)
      linka = nodes.get (a, 0)
      linkb = nodes.get (b, 0)
      nodes[a] = linka + 1
      nodes[b] = linkb + 1
      links['srcdst'][(a, b)] = 0
      links['nodelink'][(a, linka)] = (a, b)
      links['nodelink'][(b, linkb)] = (a, b)
  return links

def usage (packet_filename, links):
  with open (packet_filename, 'r') as packets:
    # expects file with parsed packet syntax used in parse-ns3-packets
    lines = packets.read ().splitlines ()[1:] # skip header
  pattern = r'(?P<recvtime>[0-9]+\.?([0-9]+)?(e[+-][0-9]+)?),' + \
            r'(?P<node>[0-9]+),' + \
            r'(?P<link>[0-9]+),' + \
            r'(?P<src>[0-9]+),' + \
            r'(?P<dst>[0-9]+),' + \
            r'(?P<size>[0-9]+)'
  for line in lines:
    m = re.match (pattern, line)
    node = int (m.group ('node'))
    link = int (m.group ('link'))
    a, b = links['nodelink'][(node, link)]
    a, b = min (a, b), max (a, b)
    links['srcdst'][(a, b)] = links['srcdst'][(a, b)] + 1
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
    usage = links['srcdst'].get ((a, b), None)
    if usage is None or usage < 1:
      continue
    pruned_links.append (link)
  yml[target_design]['ns3']['topology']['links'] = pruned_links

  old = len (design['ns3']['topology']['links'])
  new = len (pruned_links)
  percent = (new / old) 
  print ('%.1f link utilization' % (percent))

  # write out config with new links
  if percent < 1.0:
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
  parser.add_argument('-l', '--link', required = True)
  parser.add_argument('-t', '--topology', required = True)
  parser.add_argument('-c', '--configuration', required = True)
  parser.add_argument('-d', '--design', required = True)
  parser.add_argument('-o', '--override', default = False, action = 'store_true')
  args = parser.parse_args (sys.argv[1:])
  # parse topology
  links = parse (args.topology)
  # count link usage
  links = usage (args.link, links)
  # report link usage
  with open ('%s_link_usage.csv' % (args.design), 'w') as csv:
    csv.write ('node-a,node-b,usage\n')
    for a, b in links['srcdst'].keys ():
      csv.write (','.join ([str(a), str(b), str(links['srcdst'][(a,b)])]) + '\n')
  # prune links
  prune (args.configuration, args.design, links, args.override)

if __name__ == '__main__':
  main ()
