#!/usr/bin/python3

import argparse
import numpy as np
import os
import pandas as pd
import random
import sys
import yaml
import json

from astragen import (
  COMP_NODE,
  COMM_SEND_NODE,
  COMM_RECV_NODE
)

def parse_arguments ():
  parser = argparse.ArgumentParser (
    prog = 'ASTRA-nsys-converter',
    description = '',
    epilog = '')
  parser.add_argument ('--input', type = str, required = True)
  parser.add_argument ('--output', type = str, required = True)
  parser.add_argument ('--start', type = str, required = True)
  parser.add_argument ('--end', type = str, required = True)
  parser.add_argument ('--separate', default = False, action = 'store_true')
  return parser.parse_args ()

def convert_compute (nsys, node):
  node['type'] = COMP_NODE
  node['num-ops'] = nsys['sm']
  node['size'] = nsys['execute']
  node['name'] = nsys['name']
  node['duration'] = int (np.ceil ((float (nsys['end']) - float (nsys['start'])) / 1.0e3) * 1.5)
  return [node]

def convert_gather (n, tag, rank, nsys, template):
  ido = template['id']
  nodes = list ()
  # send all other nodes
  for i in range (n):
    if i == rank:
      continue
    node = dict ()
    node['id'] = ido
    ido = ido + 1
    node['type'] = COMM_SEND_NODE
    node['data-deps'] = [dep for dep in template['data-deps']]
    node['name'] = template['name'] + '_Send_' + str (i)
    node['tag'] = tag + rank * n + i
    node['dst'] = i
    node['src'] = rank
    node['size'] = nsys['execute']
    nodes.append (node)
  # recv all other nodes
  for i in range (n):
    if i == rank:
      continue
    node = dict ()
    node['id'] = ido
    ido = ido + 1
    node['type'] = COMM_RECV_NODE
    node['data-deps'] = [dep for dep in template['data-deps']]
    node['name'] = template['name'] + '_Recv_' + str (i)
    node['tag'] = tag + i * n + rank
    node['src'] = i
    node['dst'] = rank
    node['size'] = nsys['execute']
    nodes.append (node)
  tag = tag + int (n * n)
  return nodes, tag

def convert_reduce (n, tag, rank, nsys, template):
  nodes, tag = convert_gather (n, tag, rank, nsys, template)
  return nodes, tag

def convert (n, rank, nsys, workload, i, tag, start, end):
  if nsys[i]['name'] != start:
    print ('')
    sys.exit (1)
  # add most nodes
  while nsys[i]['name'] != end:
    node = {}
    node['id'] = len (workload)
    deps = []
    if node['id'] > 0:
      deps.append (node['id'] - 1)
    node['data-deps'] = deps
    node['name'] = nsys[i]['name']
    nodes = list ()
    if 'allgather' in node['name'].lower ():
      nodes, tag = convert_gather (n, tag, rank, nsys[i], node)
    elif 'allreduce' in node['name'].lower ():
      nodes, tag = convert_reduce (n, tag, rank, nsys[i], node)
    else:
      nodes = convert_compute (nsys[i], node)
    workload.extend (nodes)
    i = i + 1
  # add end node(s)
  if nsys[i]['name'] == end: 
    node = {}
    node['id'] = len (workload)
    deps = []
    if node['id'] > 0:
      deps.append (node['id'] - 1)
    node['data-deps'] = deps
    node['name'] = nsys[i]['name']
    nodes = list ()
    if 'allgather' in node['name'].lower ():
      nodes, tag = convert_gather (n, tag, rank, nsys[i], node)
    elif 'allreduce' in node['name'].lower ():
      nodes, tag = convert_reduce (n, tag, rank, nsys[i], node)
    else:
      nodes = convert_compute (nsys[i], node)
    workload.extend (nodes)
  print (f'converted {len (workload)} nodes')
  return workload, tag, i

def convert_workload (ifilename, ofilename, start, end, separate):
  data = None
  with open (ifilename, 'r') as f:
    data = json.load (f)
  workloads = list ()
  n = len (data.keys ())
  for rank, key in enumerate (data.keys ()): 
    iterations = 0
    tag = 0
    workload = list ()
    i = 0
    while i < len (data[key]):
      if data[key][i]['name'] == start:
        print (f'found start at {i}')
        if separate:
          workload, _, i = convert (n, rank, data[key], list (), i, 0, start, end)
          if not iterations > len (workloads) and rank == 0:
            workloads.append (dict ())
          workloads[iterations][rank] = dict ()
          workloads[iterations][rank]['nodes'] = workload
          iterations = iterations + 1
        else:
          workload, tag, i = convert (n, rank, data[key], workload, i, tag, start, end) 
      i = i + 1
    if not separate and rank == 0:
      workloads.append (dict ())
    workloads[0][rank] = dict ()
    workloads[0][rank]['nodes'] = workload
  print (f'got {len (workloads)} iterations with nodes {workloads[0].keys()}')
  # dump workloads
  for i, workload in enumerate (workloads):
    filename = ofilename + ('' if not separate else f'.{str(i)}') + '.yaml'
    with open (filename, 'w') as yml:
      yaml.dump ({'workload': workload}, yml, default_flow_style = False) 

def main ():
  args = parse_arguments ()
  convert_workload (args.input, args.output, args.start, args.end, args.separate)

if __name__ == '__main__': 
  main ()
