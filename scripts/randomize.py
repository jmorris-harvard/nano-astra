#!/usr/bin/python3

import argparse
import numpy as np
import os
import pandas as pd
import random
import sys
import yaml

from astragen import (
  COMP_NODE,
  COMM_SEND_NODE,
  COMM_RECV_NODE
)

# random execution trace script generator
def generate_workload (nnpus, ai, di, ops, size, nodes, output, verbose = False):
  # init output
  if verbose:
    header = '\t\t'.join (['npu %d' % i for i in range (nnpus)])
    plongest = 0
    print (header)
  tag = 0
  workload = {}
  npus = []
  free = []
  for i in range (nnpus):
    workload[i] = {}
    workload[i]['nodes'] = []
    workload[i]['remain'] = nodes
    workload[i]['backlog'] = []
    npus.append (i)
    free.append (i)
  # create
  while len (npus) > 0:
    # select random npu
    npu = random.choice (npus)

    # create node
    node = {}
    node['id'] = len (workload[npu]['nodes'])
    # select dependencies
    deps = []
    prev = [i for i in range (len(workload[npu]['nodes']))]
    while len (prev) > 0 and random.random () < di:
      dep = random.choice (prev)
      deps.append (dep)
      prev.remove (dep)
    node['data-deps'] = deps

    # roll die and select next operation
    die = random.random ()
    if len (workload[npu]['backlog']) > 0 and \
       (die > ai or len (workload[npu]['backlog']) >= workload[npu]['remain']):
      # handle recv event
      recv = workload[npu]['backlog'].pop (0)
      node['type'] = COMM_RECV_NODE
      node['size'] = recv['size']
      node['tag'] = recv['tag']
      node['src'] = recv['src']
      node['dst'] = npu
    elif  die > ai and len (free) > 1:
      # create a new send comm
      node['type'] = COMM_SEND_NODE
      node['size'] = random.randint (size[0], size[1])
      # grab new tag and increment
      node['tag'] = tag
      tag = tag + 1
      node['src'] = npu
      # select destination node (that isnt current npu)
      dst = random.choice (free)
      while dst == npu:
        dst = random.choice (free)
      node['dst'] = dst
      # create complementary recv comm
      recv = {}
      recv['size'] = node['size']
      recv['src'] = node['src']
      recv['tag'] = node['tag']
      workload[dst]['backlog'].append (recv)
    else:
      # create compute node
      node['type'] = COMP_NODE
      node['size'] = random.randint (size[0], size[1])
      node['num-ops'] = random.randint (ops[0], ops[1])
    # add node to make configure happy
    node['name'] = '%d_%d' % (node['id'], node['type'])
    # add node to tree
    workload[npu]['nodes'].append (node)

    # log node complete
    workload[npu]['remain'] = workload[npu]['remain'] - 1
    if workload[npu]['remain'] <= 0:
      # npu can no longer be set as a receiver on new send nodes
      if npu in free:
        free.remove (npu)
      if len (workload[npu]['backlog']) == 0:
        # npu has been assigned all required nodes
        npus.remove (npu)
        # destroy metadata
        workload[npu].pop('backlog')
        workload[npu].pop('remain')
    # verbose output
    if verbose:
      # get all assigned workloads
      cols = []
      longest = 0
      for i in range (nnpus):
        c = []
        for node in workload[i]['nodes']:
          if node['type'] == COMP_NODE:
            c.append ('comp')
          elif node['type'] == COMM_RECV_NODE:
            c.append ('r%d|%d' % (node['src'], node['dst']))
          elif node['type'] == COMM_SEND_NODE:
            c.append ('s%d|%d' % (node['src'], node['dst']))
          else:
            pass
        cols.append (c)
        longest = len (c) if len (c) > longest else longest
      # equalize all
      for i in range (nnpus):
        while len (cols[i]) < longest:
          cols[i].append ('')
      # zip together to create rows
      rows = ['\t\t'.join (row) for row in zip(*cols)]
      # print everything out
      # move to start
      for _ in range (plongest):
        print ('\033[F', end = '')
      for row in rows:
        print (row)
      plongest = longest

  with open (output, 'w') as yml:
    yaml.dump ({'workload': workload}, yml, default_flow_style = False)

def main ():
  parser = argparse.ArgumentParser (
    prog = 'ASTRA-workload-generator',
    description = '',
    epilog = '')
  parser.add_argument ('-n', '--npus', type = int, required = True)
  parser.add_argument ('-a', '--arithmetic', type = float, required = True)
  parser.add_argument ('-d', '--dependency', type = float, required = True)
  parser.add_argument ('-o', '--output', type = str, required = True)
  parser.add_argument ('-v', '--verbose', action = 'store_true', default = False)
  parser.add_argument ('--min_ops', type = int, default = int (1e3))
  parser.add_argument ('--max_ops', type = int, default = int (1e12))
  parser.add_argument ('--min_size', type = int, default = int (1e3))
  parser.add_argument ('--max_size', type = int, default = int (1e12))
  parser.add_argument ('--nodes', type = int, default = int (10))
  parser.add_argument ('--seed', type = int, default = None)
  args = parser.parse_args (sys.argv[1:])
  # assert args
  assert args.arithmetic > 0.0 and args.arithmetic < 1.0,\
         '-a,--arithmetic FLOAT should be in (0.0, 1.0)'
  assert args.dependency > 0.0 and args.dependency < 1.0,\
         '-d,--dependency FLOAT should be in (0.0, 1.0)'
  random.seed (args.seed)
  generate_workload (
    args.npus,
    args.arithmetic,
    args.dependency,
    (args.min_ops, args.max_ops),
    (args.min_size, args.max_size),
    args.nodes,
    args.output,
    args.verbose
  )

if __name__ == '__main__':
  main ()
