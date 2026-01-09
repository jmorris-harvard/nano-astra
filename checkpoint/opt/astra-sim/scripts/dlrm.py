#!/usr/bin/python3

import argparse
import numpy as np
import os
import pandas as pd
import random
import re
import sys
import yaml

from astragen import (
  COMP_NODE,
  COMM_SEND_NODE,
  COMM_RECV_NODE
)

# random execution trace script generator
def generate_workload (nnpus, depth, width, tables, size, samples, batch, ebatch, dim, pool, output, verbose = False):
  # init output
  if verbose:
    header = '\t\t'.join (['npu %d' % i for i in range (nnpus)])
    plongest = 0
    print (header)
  tag = 0
  workload = {}
  inc = []
  for i in range (nnpus):
    workload[i] = {}
    workload[i]['nodes'] = []
    workload[i]['tables'] = 0
  
  # all gpus:
  # --> distribute tables among npus in round robin
  i = 0
  while tables > 0:
    if i not in inc:
      inc.append (i)
    workload[i]['tables'] = workload[i]['tables'] + 1
    i = (i + 1) % nnpus
    tables = tables - 1

  # --> assign dummy workloads to not included
  if len (inc) < len (workload.keys ()):
    tnode = {}
    tnode['id'] = 0
    tnode['name'] = '0_dummy_comp'
    tnode['data-deps'] = []
    tnode['type'] = COMP_NODE
    tnode['size'] = 8
    tnode['num-ops'] = 0
    for npu in workload.keys ():
      if npu not in inc:
        node = {}
        for key in tnode.keys ():
          if key == 'data-deps':
            node[key] = []
            for d in tnode[key]:
              node[key].append (d)
          else:
            node[key] = tnode[key]
        workload[npu]['nodes'].append (node)

  # --> do mlp bot using dense inputs
  # ----> assumes all layers in mlp have the same width
  # ----> assumes parameters are each 2 bytes (16-bit)
  # ----> assumes constant sparse ids sample length
  # compute operation characteristics
  # input size + weight size + output size
  # divided by number npus for hybrid compute
  t_size = 2 * ((width * batch) + (width * width * depth) + (width * depth))
  # ops per intermediate and outputs
  # divided by number npus for hybrid compute
  t_ops = depth * batch * (width * 2 * width)
  # Jalil
  print ('mlp bytes (per device): ', t_size)
  print ('mlp ops (per device): ', t_ops)
  # Morris
  # create template node
  tnode = {}
  tnode['id'] = 0
  tnode['name'] = '%d_mlp_bot'
  tnode['data-deps'] = []
  tnode['type'] = COMP_NODE
  tnode['size'] = t_size
  tnode['num-ops'] = t_ops
  # add template node for each table
  for npu in inc:
    # pool tables
    node = {}
    for key in tnode.keys ():
      if key == 'data-deps':
        node[key] = []
        for d in tnode[key]:
          node[key].append (d)
      else:
        node[key] = tnode[key]
    node['name'] = node['name'] % node['id']
    workload[npu]['nodes'].append (node)

  # --> do embedding table search based on local table
  # ----> hash function to table lookup
  # --> do all to all
  # send to all nodes with a table
  # samples --> number lookups
  # access full table 
  # perform op (concat/addition/etc)
  # removed table size --> not accounted
  t_size = 2 * dim * samples * ebatch * int (len (inc))
  t_ops = dim * samples * ebatch
  # Jalil
  print ('lookup bytes (per device): ', t_size * workload[npu]['tables'])
  # Morris
  tnode = {}
  tnode['id'] = 0
  tnode['name'] = '%d_%d_emb'
  tnode['data-deps'] = []
  tnode['type'] = COMP_NODE
  tnode['size'] = t_size
  tnode['num-ops'] = t_ops

  c_size = 2 * dim * ebatch
  # Jalil
  print ('bytes to send (per device): ', c_size * workload[npu]['tables'])
  # Morris
  cnode = {}
  cnode['id'] = 0
  cnode['name'] = '%s_send_emb'
  cnode['data-deps'] = []
  cnode['type'] = COMM_SEND_NODE
  cnode['size'] = c_size
  cnode['src'] = 0
  cnode['dst'] = 0
  cnode['tag'] = 0

  recvs = {} 
  for npu in inc:
    recvs[npu] = []

  tag = 0
  for npu in inc:
    # pool tables
    t_nodes = int (workload[npu]['tables'] / pool)
    t_remain = workload[npu]['tables'] % pool
    s_nodes = [pool for p in range (t_nodes)]
    if t_remain > 0:
      s_nodes.append (t_remain)
    for i, p_size in enumerate (s_nodes):
      offset = len (workload[npu]['nodes'])
      # do lookup
      node = {}
      for key in tnode.keys ():
        if key == 'data-deps':
          node[key] = []
          for d in tnode[key]:
            node[key].append (d)
        else:
          node[key] = tnode[key]
      node['id'] = offset
      node['size'] = node['size'] * p_size
      node['num-ops'] = node['num-ops'] * p_size
      node['name'] = node['name'] % (offset, i)
      workload[npu]['nodes'].append (node)
      # send out result
      j = 1
      for dst in inc:
        if dst == npu:
          continue
        snode = {}
        for key in cnode.keys ():
          if key == 'data-deps':
            snode[key] = []
            for d in cnode[key]:
              snode[key].append (d)
          else:
            snode[key] = cnode[key]
        snode['id'] = offset + j
        snode['name'] = snode['name'] % (offset + j)
        snode['src'] = npu
        snode['size'] = snode['size'] * p_size
        snode['dst'] = dst
        snode['tag'] = tag
        snode['data-deps'] = [offset]
        workload[npu]['nodes'].append (snode)
        j = j + 1
        tag = tag + 1

        dnode = {}
        for key in snode.keys ():
          dnode[key] = snode[key]
        dnode['name'] = '%d_recv_emb'
        dnode['type'] = COMM_RECV_NODE
        dnode['data-deps'] = []
        recvs[dst].append (dnode)

  # --> do recvs
  # --> do feature extraction and mlp top
  t_size = 2 * ((width * batch) + (width * width * depth) + (width * depth))
  # ops per intermediate and outputs
  # divided by number npus for hybrid compute
  t_ops = depth * batch * (width * 2 * width)
  tnode = {}
  tnode['id'] = 0
  tnode['name'] = '%d_mlp_top'
  tnode['data-deps'] = []
  tnode['type'] = COMP_NODE
  tnode['size'] = t_size
  tnode['num-ops'] = t_ops
  for npu in inc:
    t_nodes = int (workload[npu]['tables'] / pool)
    t_remain = workload[npu]['tables'] % pool
    s_nodes = [pool for p in range (t_nodes)]
    if t_remain > 0:
      s_nodes.append (t_remain)
    # get all other table data
    deps = []
    offset = len (workload[npu]['nodes'])
    for i, node in enumerate (recvs[npu]):
      node['id'] = offset + i
      node['name'] = node['name'] % (offset + i)
      # node['data-deps'] = 
      workload[npu]['nodes'].append (node)
      deps.append (offset + i)
    offset = len (workload[npu]['nodes'])
    node = {}
    for key in tnode.keys ():
      if key == 'data-deps':
        node[key] = []
        for d in tnode[key]:
          node[key].append (d)
      else:
        node[key] = tnode[key]
    node['id'] = offset
    node['name'] = node['name'] % (offset)
    node['data-deps'] = [d for d in deps]
    # add mlp bot as a dependency
    node['data-deps'].append (0)
    # add embeddings as a dependency
    for j in range (len (s_nodes)):
      node['data-deps'].append (j + 1)
    workload[npu]['nodes'].append (node)

  with open (output, 'w') as yml:
    yaml.dump ({'workload': workload}, yml, default_flow_style = False)

def main ():
  parser = argparse.ArgumentParser (
    prog = 'ASTRA-dlrm-workload-generator',
    description = '',
    epilog = '')
  parser.add_argument ('-n', '--npus', type = int, required = True)
  parser.add_argument ('-o', '--output', type = str, required = True)

  parser.add_argument ('--depth', default = int (8))
  parser.add_argument ('--width', default = int (8))
  parser.add_argument ('--tables', default = int (1))
  parser.add_argument ('--size', default = int (128))
  parser.add_argument ('--samples', default = int (4))
  parser.add_argument ('--batch', default = int (16))
  parser.add_argument ('--ebatch', default = int (16))
  parser.add_argument ('--dim', default = int (16))
  parser.add_argument ('--pool', default = int (1))

  parser.add_argument ('-v', '--verbose', action = 'store_true', default = False)
  parser.add_argument ('--seed', type = int, default = None)
  args = parser.parse_args (sys.argv[1:])
  random.seed (args.seed)
  generate_workload (
    int (args.npus),
    int (args.depth),
    int (args.width),
    int (args.tables),
    int (args.size),
    int (args.samples),
    int (args.batch),
    int (args.ebatch),
    int (args.dim),
    int (args.pool),
    args.output,
    args.verbose
  )

if __name__ == '__main__':
  main ()
