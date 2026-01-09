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

def init_workload (nnpus):
  workload = {}
  for i in range (nnpus):
    workload[i] = {}
    workload[i]['nodes'] = []
  return workload

def init_deps (nnpus):
  deps = {}
  for i in range (nnpus):
    deps[i]= []
  return deps

def compute (nnpus, size, flops, workload = None, deps = None, basename = 'compute'):
  cnode = {}
  if workload == None:
    workload = init_workload (nnpus)
  if deps == None:
    deps = init_deps (nnpus)

  cnode = {}
  cnode['id'] = None
  cnode['name'] = basename + '_%s'
  cnode['data-deps'] = None
  cnode['type'] = COMP_NODE
  cnode['size'] = size
  cnode['num-ops'] = flops

  for i in range (nnpus):
    cnode_t = {}
    for key in cnode.keys ():
      cnode_t[key] = cnode[key]
    cnode_t['id'] = len (workload[i]['nodes'])
    cnode_t['data-deps'] = [dep for dep in deps[i]]
    workload[i]['nodes'].append (cnode_t)
    deps[i] = [cnode_t['id']]

  return workload, deps

def all2all_default (nnpus, size, workload, deps, basename = 'all_2_all'):
  snode = {}
  snode['id'] = None
  snode['name'] = basename + '_%s_send_%s'
  snode['data-deps'] = None
  snode['type'] = COMM_SEND_NODE
  snode['size'] = size
  snode['src'] = None
  snode['dst'] = None
  snode['tag'] = None

  tag = max ([len (workload[i]['nodes']) for i in range (nnpus)]) + 1
  tagmap = {}
  for src in range (nnpus):
    depsn = []
    for dst in range (nnpus):
      if src == dst:
        continue
      snode_t = {}
      for key in snode.keys ():
        snode_t[key] = snode[key]
      snode_t['id'] = len (workload[src]['nodes'])
      snode_t['name'] = snode_t['name'] % (snode_t['id'], dst)
      snode_t['data-deps'] = [dep for dep in deps[src]]
      snode_t['src'] = src
      snode_t['dst'] = dst
      snode_t['tag'] = tag
      tag = tag + 1
      workload[src]['nodes'].append (snode_t)
      tagmap[(src, dst)] = snode_t['tag']
      depsn.append (snode_t['id'])
    deps[src] = depsn

  rnode = {}
  rnode['id'] = None
  rnode['name'] = 'all_2_all_%s_recv_%s'
  rnode['data-deps'] = None
  rnode['type'] = COMM_RECV_NODE
  rnode['size'] = size
  rnode['src'] = None
  rnode['dst'] = None
  rnode['tag'] = None
  for dst in range (nnpus):
    depsn = []
    for src in range (nnpus):
      if dst == src:
        continue
      rnode_t = {}
      for key in rnode.keys ():
        rnode_t[key] = rnode[key]
      rnode_t['id'] = len (workload[dst]['nodes'])
      rnode_t['name'] = rnode_t['name'] % (rnode_t['id'], src)
      rnode_t['data-deps'] = [dep for dep in deps[dst]]
      rnode_t['src'] = src
      rnode_t['dst'] = dst
      rnode_t['tag'] = tagmap[(src, dst)]
      workload[dst]['nodes'].append (rnode_t)
      depsn.append (rnode_t['id'])
    deps[dst] = depsn

  return workload, deps
      
def all2all (nnpus, size, workload = None, deps = None, algorithm = 'default'):
  if workload == None:
    workload = init_workload (nnpus)
  if deps == None:
    deps = init_deps (nnpus)

  if algorithm == 'default':
    return all2all_default (nnpus, size, workload, deps)

def allreduce_default (nnpus, size, workload, deps, basename = 'all_reduce'):
  workload, deps = all2all_default (nnpus, size, workload, deps, basename)
  return compute (nnpus, size, size, workload, deps, basename)

def allreduce (nnpus, size, workload = None, deps = None, algorithm = 'default'):
  if workload == None:
    workload = init_workload (nnpus)
  if deps == None:
    deps = init_deps (nnpus)

  if algorithm == 'default':
    return allreduce_default (nnpus, size, workload, deps)

def allgather_default (nnpus, size, workload, deps, basename = 'all_gather'):
  return all2all_default (nnpus, int (size / nnpus), workload, deps, basename)

def allgather (nnpus, size, workload = None, deps = None, algorithm = 'default'):
  if workload == None:
    workload = init_workload (nnpus)
  if deps == None:
    deps = init_deps (nnpus)

  if algorithm == 'default':
    return allgather_default (nnpus, size, workload, deps, basename = 'all_gather')

def mlp (nnpus, width, batch, layers, workload = None, deps = None, comm_algorithm = 'default'):
  if workload == None:
    workload = init_workload (nnpus)
  if deps == None:
    deps = init_deps (nnpus)

  # fwd 
  for layer in range (layers):
    workload, deps = compute (
        nnpus, 
        2 * width * batch, 
        2 * batch * width * width, 
        workload,
        deps,
        'mlp_fwd_%d' % (layer)
    )
    workload, deps = allgather (
        nnpus,
        2 * width * batch,
        workload,
        deps,
        comm_algorithm
    )

  # bwd
  for layer in range (layers):
    workload, deps = compute (
        nnpus,
        2 * width,
        2 * width * width,
        workload,
        deps,
        'mlp_bwd_%d' % (len (layers) - layer - 1)
    )
    workload, deps = allreduce (
        nnpus,
        2 * width,
        workload,
        deps,
        comm_algorithm
    )

  return workload, deps

