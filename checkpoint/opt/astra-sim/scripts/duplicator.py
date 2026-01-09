#!/usr/bin/python3

import argparse
import numpy as np
import os
import pandas as pd
import random
import re
import sys
import yaml

import etgenerate
from astragen import (
  ChakraNode,
  ChakraAttr,
  GlobalMetadata,
  decode,
  encode,
  openFile,
  INVALID_NODE,
  COMP_NODE,
  COMM_COLL_NODE,
  ALL_REDUCE,
  COMM_SEND_NODE,
  COMM_RECV_NODE
)

def stg_pipeline_dependencies (nodes, dups, idoffset):
  # ensure first nodes in subsequent jobs are dependent on completion of previous nodes
  start_fwd_nodes = []
  end_fwd_nodes = []
  start_bwd_nodes = []
  end_bwd_nodes = []
  for node in nodes:
    # grab start nodes
    pattern = r'in_emb_y@[0-9]+_COMP(_[0-9]+)*'
    if re.match (pattern, node.name) is not None:
      start_fwd_nodes.append (int (node.id))
    
    pattern = r'stack_[0-9]+_mha_q@[0-9]+_COMP(_[0-9]+)*'
    if re.match (pattern, node.name) is not None:
      start_fwd_nodes.append (int (node.id))
    
    pattern = r'stack_[0-9]+_mha_k@[0-9]+_COMP(_[0-9]+)*'
    if re.match (pattern, node.name) is not None:
      start_fwd_nodes.append (int (node.id))
    
    pattern = r'stack_[0-9]+_mha_v@[0-9]+_COMP(_[0-9]+)*'
    if re.match (pattern, node.name) is not None:
      start_fwd_nodes.append (int (node.id))

    pattern = r'shadow_stack_[0-9]+_mha_d_x@[0-9]+_Y_RECV(_[0-9]+)*'
    if re.match (pattern, node.name) is not None:
      start_bwd_nodes.append (int (node.id))

    # grab end nodes 
    pattern = r'stack_[0-9]+_ffn_norm@[0-9]+_Y_SEND(_[0-9]+)*'
    if re.match (pattern, node.name) is not None:
      end_fwd_nodes.append (int (node.id))

    pattern = r'stack_[0-9]+_mha_d_x@[0-9]+_Y_SEND(_[0-9]+)*'
    if re.match (pattern, node.name) is not None:
      end_bwd_nodes.append (int (node.id))

    pattern = r'out_emb_dx@[0-9]+_Y_SEND(_[0-9]+)*'
    if re.match (pattern, node.name) is not None:
      end_bwd_nodes.append (int (node.id))

  if len (start_fwd_nodes) == 0 or \
     len (end_fwd_nodes) == 0 or \
     len (end_bwd_nodes) == 0:
    print ('could not find all checkpoint nodes (stg_pipeline) format is likely wrong')
    print (start_fwd_nodes)
    print (end_fwd_nodes)
    print (end_bwd_nodes)
    sys.exit ()

  dependency_dict = dict ()
  for start in start_fwd_nodes:
    for i in range (1, dups):
      dependency_dict[start + (i * idoffset)] = list ()
      for dep in end_fwd_nodes:
        dependency_dict[start + (i * idoffset)].append (dep + ((i - 1) * idoffset))

  for start in start_bwd_nodes:
    for i in range (1, dups):
      dependency_dict.setdefault (start + (i * idoffset), list ())
      for dep in end_bwd_nodes:
        dependency_dict[start + (i * idoffset)].append (dep + ((i - 1) * idoffset))
  return dependency_dict

def duplicateWorkload (npu, execution_trace_filename, dups, dependencies = []):
  execution_trace = openFile (execution_trace_filename)
  # read metadata node
  metadata = GlobalMetadata ()
  decode (execution_trace, metadata)
  # get baseline job
  nodes = list ()
  node = ChakraNode ()
  while decode (execution_trace, node):
    if node.type == INVALID_NODE:
      continue
    nodes.append (node)
    node = ChakraNode ()
  # iterate through all nodes to get the max value
  # set as offset for future nodes
  idoffset = 0
  tagoffset = 0
  for node in nodes:
    if int (node.id) > idoffset:
      idoffset = int (node.id)
    if node.type == COMM_SEND_NODE or node.type == COMM_RECV_NODE:
      for attr in node.attr:
        if attr.name == 'comm_tag':
          if int (attr.int32_val) > tagoffset:
            tagoffset = int (attr.int32_val)
  idoffset = idoffset + 1
  # return tag offset to be used to update later
  tagoffset = tagoffset + 1
  # get dependencies
  dependency_dict = dict ()
  for dependency in dependencies:
    if dependency.lower () == 'stg_pipeline'.lower ():
      stg_dependencies = stg_pipeline_dependencies (nodes, dups, idoffset)
      for key in stg_dependencies.keys ():
        dependency_dict.setdefault (key, [])
        dependency_dict[key].extend (stg_dependencies[key])
    else:
      print ('unrecognized dependency (%s)' % (dependency))
      sys.exit ()
  # iterate through all nodes and update:
  # -- name
  # -- id
  # -- data_deps
  output = []
  tags = []
  ctr = 0
  for node in nodes:
    for i in range (dups):
      dup = dict ()
      # add on required values
      # increment id
      dup['id'] = int (node.id) + (i * idoffset)
      # add suffix to name
      dup['name'] = node.name + '_' + str (i)
      dup['type'] = node.type
      # type insurance
      if dup['type'] == COMM_SEND_NODE:
        dup['src'] = npu
      if dup['type'] == COMM_RECV_NODE:
        dup['dst'] = npu
      # increment dependencies
      dup['data-deps'] = []
      for dep in node.data_deps:
        dup['data-deps'].append (dep + (i * idoffset))
      # add additional dependencies when necessary
      if dup['id'] in dependency_dict.keys ():
        # append dependency
        dup['data-deps'].extend (dependency_dict[dup['id']])
      dup['duration'] = node.duration_micros
      # add on attributes
      for attr in node.attr:
        if attr.name == 'num_ops':
          dup['num-ops'] = int (attr.int64_val)
        elif attr.name == 'comm_type':
          dup['collective'] = int (attr.int64_val)
        elif attr.name == 'comm_priority':
          dup['priority'] = int (attr.int32_val)
        elif attr.name == 'comm_src':
          dup['src'] = int (attr.int32_val)
        elif attr.name == 'comm_dst':
          dup['dst'] = int (attr.int32_val)
        # increment send data tag
        elif attr.name == 'comm_tag':
          dup['tag'] = int (attr.int32_val)
          # track multiplier and where in output to update later
          tags.append ((ctr, i))
        elif attr.name == 'tensor_size':
          dup['size'] = int (attr.uint64_val)
        elif attr.name == 'comm_size':
          dup['size'] = int (attr.int64_val)
        elif attr.name == 'is_cpu_op':
          pass
        else:
          print ('unrecognized node attribute (%s)' % (attr.name))
          sys.exit ()
      # add duplicate to output
      output.append (dup)
      ctr = ctr + 1
  # sort output by id
  # output = sorted (output, key = lambda x:x['id'])
  # print (output)
  return output, tags, tagoffset

def main ():
  parser = argparse.ArgumentParser (
    prog = 'ASTRA-workload-generator',
    description = '',
    epilog = '')
  parser.add_argument ('-f',
                       '--filename',
                       type = str,
                       required = True,
                       help = 'Execution trace filename prefix')
  parser.add_argument ('-n',
                       '--duplicates',
                       type = int,
                       required = True,
                       help = 'Number times to duplicate the workload')
  parser.add_argument ('-i',
                       '--inplace',
                       default = False,
                       action = 'store_true',
                       help = 'Output execution traces as opposed to json')
  parser.add_argument ('-d',
                       '--format',
                       nargs = '+',
                       default = [],
                       help = 'Assume job dependencies have one or more of these properties [ stg_pipeline ]')
  args = parser.parse_args (sys.argv[1:])
  files = dict ()
  head, tail = os.path.split (args.filename)
  pattern = tail + r'\.(?P<npu>[0-9]+)\.et'
  for filename in os.listdir (head):
    m = re.match (pattern, filename)
    if m is not None:
      npu = int (m.group ('npu'))
      files[npu] = os.path.join (head, filename)
  if len (files.keys ()) == 0:
    print ('no files found with pattern %s.<npu>.et ' % (args.filename))
    sys.exit ()
  
  workload = dict ()
  tags = dict ()

  # extract all workload data
  for npu in files.keys ():
    workload[npu] = dict ()
    tags[npu] = dict ()
    workload[npu]['nodes'], tags[npu]['nodes'], tags[npu]['offset'] = duplicateWorkload (npu, files[npu], args.duplicates, args.format)

  # update tags globally
  tagoffset = max ([tags[npu]['offset'] for npu in tags.keys ()])
  for npu in tags.keys ():
    for loc, multiplier in tags[npu]['nodes']:
      workload[npu]['nodes'][loc]['tag'] = workload[npu]['nodes'][loc]['tag'] + multiplier * tagoffset

  # write out
  if not args.inplace:
    with open (args.filename + '.yaml', 'w') as yml:
      yaml.dump ({'workload': workload}, yml, default_flow_style = False)
  else:
    for npu in files.keys ():
      with open (files[npu], 'wb') as et:
        encode (et, GlobalMetadata (version = '0.0.4'))
        etgenerate.generate (npu, workload[npu]['nodes'], et)

if __name__ == '__main__':
  main ()
