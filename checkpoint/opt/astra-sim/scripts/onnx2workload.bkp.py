#!/usr/bin/python3

import argparse
import numpy as np
import os
import pandas as pd
import random
import sys
import re
import yaml
import json

from astragen import (
  COMP_NODE,
  COMM_SEND_NODE,
  COMM_RECV_NODE
)

def parse_arguments ():
  parser = argparse.ArgumentParser (
    prog = 'ASTRA-onnx-converter',
    description = '',
    epilog = '')
  parser.add_argument ('--input', type = str, required = True, help = 'input onnx network name')
  parser.add_argument ('--dir', type = str, required = True, help = 'onnx networks location')
  parser.add_argument ('--output', type = str, required = True, help = 'output name (as .yaml)')
  parser.add_argument ('--batch', type = int, required = False, default = 1, help = 'input batch size')
  parser.add_argument ('--sequence', type = int, required = False, default = 1, help = 'number tokens to execute')
  parser.add_argument ('--split_sends', type = int, required = False, default = 1, help = 'number time to split send ops (improves utilization)')
  parser.add_argument ('--verbose', action = 'store_true', required = False, default = False, help = 'generate verbose output')
  return parser.parse_args ()

def convert (ifilename, batch, rank, n, split_sends, sequence, verbose): 
  onnx = None
  with open (ifilename, 'r') as ifile:
    onnx = json.load (ifile)
  shapes = dict () # output name to shape tuple
  mapping = dict () # layer name to node if exists
  workload = list ()
  tag = 0
  # set up inputs
  for i in onnx['inputs'].values ():
    # update inputs to reflect batch size and add to shape map
    shapes[i['name']] = tuple ([int (s) if int (s) > -1 else batch for s in i['shape']])
  for i in range (len (onnx['layers'])):
    layer = onnx['layers'][str (i)]
    # computational
    if layer['type'] == 'Mul': # elementwise multiplication
      mapping[i] = dict ()
      mapping[i]['deps'] = None
      node = {
        'id': len (workload),
        'name': layer['id'],
        'data-deps': [
          mapping[dep]['id'] for dep in layer['dependencies'] if dep > -1 and mapping[dep]['id'] > -1
        ],
        'duration': 1, # just a placeholder value
        'type': COMP_NODE,
      }
      # add non compute layer dependencies
      for dep in layer['dependencies']:
        if dep > -1 and mapping[dep]['deps'] is not None:
          node['data-deps'].extend ([d for d in mapping[dep]['deps'] if d not in node['data-deps']])
      node['num-ops'] = int (np.prod (shapes[layer['inputs'][0]]))
      node['size'] = 2 * node['num-ops']
      mapping[i]['id'] = node['id']
      shapes[layer['output']] = shapes[layer['inputs'][0]]
      workload.append (node)
    elif layer['type'] == 'Pow': # elementwise power (assume square)
      mapping[i] = dict ()
      mapping[i]['deps'] = None
      node = {
        'id': len (workload),
        'name': layer['id'],
        'data-deps': [
          mapping[dep]['id'] for dep in layer['dependencies'] if dep > -1 and mapping[dep]['id'] > -1
        ],
        'duration': 1, # just a placeholder value
        'type': COMP_NODE,
      }
      # add non compute layer dependencies
      for dep in layer['dependencies']:
        if dep > -1 and mapping[dep]['deps'] is not None:
          node['data-deps'].extend ([d for d in mapping[dep]['deps'] if d not in node['data-deps']])
      node['num-ops'] = int (np.prod (shapes[layer['inputs'][0]]))
      node['size'] = 2 * node['num-ops']
      mapping[i]['id'] = node['id']
      shapes[layer['output']] = shapes[layer['inputs'][0]]
      workload.append (node)
    elif layer['type'] == 'ReduceMean': # computes mean across axis
      mapping[i] = dict ()
      mapping[i]['deps'] = None
      node = {
        'id': len (workload),
        'name': layer['id'],
        'data-deps': [
          mapping[dep]['id'] for dep in layer['dependencies'] if dep > -1 and mapping[dep]['id'] > -1
        ],
        'duration': 1, # just a placeholder value
        'type': COMP_NODE,
      }
      # add non compute layer dependencies
      for dep in layer['dependencies']:
        if dep > -1 and mapping[dep]['deps'] is not None:
          node['data-deps'].extend ([d for d in mapping[dep]['deps'] if d not in node['data-deps']])
      axis = int (layer['details']['axes'][0]) - 1
      oshape = None
      if layer['details']['keep_dims']:
        oshape = tuple ([s if j != axis else 1 for j, s in enumerate (shapes[layer['inputs'][0]])])
      else:
        oshape = tuple ([s for j, s in enumerate (shapes[layer['inputs'][0]]) if j != axis])
      try:
        ops = shapes[layer['inputs'][0]][axis]
      except:
        print (layer['inputs'][0])
        print (shapes[layer['inputs'][0]])
        print (axis)
        sys.exit (1)
      node['num-ops'] = int (np.prod (oshape) * (ops + 1))
      node['size'] = int (np.prod (shapes[layer['inputs'][0]]))
      mapping[i]['id'] = node['id']
      shapes[layer['output']] = oshape
      workload.append (node)
    elif layer['type'] == 'Add': # elementwise addition
      mapping[i] = dict ()
      mapping[i]['deps'] = None
      node = {
        'id': len (workload),
        'name': layer['id'],
        'data-deps': [
          mapping[dep]['id'] for dep in layer['dependencies'] if dep > -1 and mapping[dep]['id'] > -1
        ],
        'duration': 1, # just a placeholder value
        'type': COMP_NODE,
      }
      # add non compute layer dependencies
      for dep in layer['dependencies']:
        if dep > -1 and mapping[dep]['deps'] is not None:
          node['data-deps'].extend ([d for d in mapping[dep]['deps'] if d not in node['data-deps']])
      node['num-ops'] = int (np.prod (shapes[layer['inputs'][0]]))
      node['size'] = 2 * node['num-ops']
      mapping[i]['id'] = node['id']
      shapes[layer['output']] = shapes[layer['inputs'][0]]
      workload.append (node)
    elif layer['type'] == 'Sqrt': # elementwise square root
      mapping[i] = dict ()
      mapping[i]['deps'] = None
      node = {
        'id': len (workload),
        'name': layer['id'],
        'data-deps': [
          mapping[dep]['id'] for dep in layer['dependencies'] if dep > -1 and mapping[dep]['id']
        ],
        'duration': 1, # just a placeholder value
        'type': COMP_NODE,
      }
      # add non compute layer dependencies
      for dep in layer['dependencies']:
        if dep > -1 and mapping[dep]['deps'] is not None:
          node['data-deps'].extend ([d for d in mapping[dep]['deps'] if d not in node['data-deps']])
      node['num-ops'] = int (np.prod (shapes[layer['inputs'][0]]))
      node['size'] = 2 * node['num-ops']
      mapping[i]['id'] = node['id']
      shapes[layer['output']] = shapes[layer['inputs'][0]]
      workload.append (node)
    elif layer['type'] == 'Div': # elementwise division
      mapping[i] = dict ()
      mapping[i]['deps'] = None
      node = {
        'id': len (workload),
        'name': layer['id'],
        'data-deps': [
          mapping[dep]['id'] for dep in layer['dependencies'] if dep > -1 and mapping[dep]['id'] > -1
        ],
        'duration': 1, # just a placeholder value
        'type': COMP_NODE,
      }
      # add non compute layer dependencies
      for dep in layer['dependencies']:
        if dep > -1 and mapping[dep]['deps'] is not None:
          node['data-deps'].extend ([d for d in mapping[dep]['deps'] if d not in node['data-deps']])
      node['num-ops'] = int (np.prod (shapes[layer['inputs'][0]]))
      node['size'] = 2 * node['num-ops']
      mapping[i]['id'] = node['id']
      shapes[layer['output']] = shapes[layer['inputs'][0]]
      workload.append (node)
    elif layer['type'] == 'Gemm': # D = alpha * A * B + beta * C
      mapping[i] = dict ()
      mapping[i]['deps'] = None
      node = {
        'id': len (workload),
        'name': layer['id'],
        'data-deps': [
          mapping[dep]['id'] for dep in layer['dependencies'] if dep > -1 and mapping[dep]['id'] > -1
        ],
        'duration': 1, # just a placeholder value
        'type': COMP_NODE,
      }
      # add non compute layer dependencies
      for dep in layer['dependencies']:
        if dep > -1 and mapping[dep]['deps'] is not None:
          node['data-deps'].extend ([d for d in mapping[dep]['deps'] if d not in node['data-deps']])
      Ashape = shapes[layer['inputs'][0]]
      Bshape = shapes[layer['inputs'][1]]
      if layer['details']['op0'] == 'MatrixOperation.TRANSPOSE':
        Ashape = tuple ([Ashape[1], Ashape[0]])
      if layer['details']['op1'] == 'MatrixOperation.TRANSPOSE':
        Bshape = tuple ([Bshape[1], Bshape[0]])
      try:
        M, K, N = Ashape[0], Ashape[1], Bshape[1]
      except:
        for k, v in shapes.items ():
          print (f'{k}\n{v}\n')
        print (Ashape)
        print (Bshape)
        sys.exit ()
      node['num-ops'] = int (2 * M * N * K + M * N)
      node['size'] = int (M * K + K * N + 2 * M * N)
      mapping[i]['id'] = node['id']
      shapes[layer['output']] = tuple ([M, N])
      workload.append (node)
    elif layer['type'] == 'PLUGIN_V2': # other computations/collectives TODO
      mapping[i] = dict ()
      mapping[i]['deps'] = None
      template = {
        'name': layer['id'],
        'data-deps': [
          mapping[dep]['id'] for dep in layer['dependencies'] if dep > -1 and mapping[dep]['id'] > -1
        ],
        'duration': 1, # just a placeholder value
      }
      # add non compute layer dependencies
      for dep in layer['dependencies']:
        if dep > -1 and mapping[dep]['deps'] is not None:
          template['data-deps'].extend ([d for d in mapping[dep]['deps'] if d not in template['data-deps']])
      # collectives
      if 'AllReduce' in layer['id']:
        csize = int (np.prod(shapes[layer['inputs'][0]])) * 2
        csize = int (csize / split_sends)
        deps = list ()
        # do all sends
        for j in range (n):
          if j == rank:
            continue
          for k in range (split_sends):
            node = {
              'id': len (workload),
              'name': template['name'] + f'_send_{j}',
              'data-deps': [dep for dep in template['data-deps']],
              'duration': 1,
              'type': COMM_SEND_NODE,
              'tag': tag + k * n * n + rank * n + j,
              'dst': j,
              'src': rank,
              'size': csize
            }
            workload.append (node)
            deps.append (node['id'])
        # do all recvs
        for j in range (n):
          if j == rank:
            continue
          for k in range (split_sends):
            node = {
              'id': len (workload),
              'name': template['name'] + f'_recv_{j}',
              'data-deps': [dep for dep in template['data-deps']],
              'duration': 1,
              'type': COMM_RECV_NODE,
              'tag': tag + k * n * n + j * n + rank,
              'dst': rank,
              'src': j,
              'size': csize
            }
            workload.append (node)
            deps.append (node['id'])
        tag = tag + split_sends * n * n + n * n
        shapes[layer['output']] = tuple (shapes[layer['inputs'][0]])
        mapping[i]['id'] = -1
        mapping[i]['deps'] = deps
      elif 'AllGather' in layer['id']:
        csize = int (np.prod(shapes[layer['inputs'][0]])) * 2 # assume float16
        csize = int (csize / split_sends)
        deps = list ()
        # do all sends
        for j in range (n):
          if j == rank:
            continue
          for k in range (split_sends):
            node = {
              'id': len (workload),
              'name': template['name'] + f'_send_{j}',
              'data-deps': [dep for dep in template['data-deps']],
              'duration': 1,
              'type': COMM_SEND_NODE,
              'tag': tag + k * n * n + rank * n + j,
              'dst': j,
              'src': rank,
              'size': csize
            }
            workload.append (node)
            deps.append (node['id'])
        # do all recvs
        for j in range (n):
          if j == rank:
            continue
          for k in range (split_sends):
            node = {
              'id': len (workload),
              'name': template['name'] + f'_recv_{j}',
              'data-deps': [dep for dep in template['data-deps']],
              'duration': 1,
              'type': COMM_RECV_NODE,
              'tag': tag + k * n * n + j * n + rank,
              'dst': rank,
              'src': j,
              'size': csize
            }
            workload.append (node)
            deps.append (node['id'])
        tag = tag + split_sends * n * n + n * n
        shapes[layer['output']] = tuple ([shapes[layer['inputs'][0]][0], n * shapes[layer['inputs'][0]][1]])
        mapping[i]['id'] = -1
        mapping[i]['deps'] = deps
      elif 'Attention' in layer['id']:
        template = {
          'id': len (workload),
          'name': layer['id'],
          'data-deps': [
            mapping[dep]['id'] for dep in layer['dependencies'] if dep > -1 and mapping[dep]['id'] > -1
          ],
          'duration': 1, # just a placeholder value
          'type': COMP_NODE,
        }
        # add non compute layer dependencies
        for dep in layer['dependencies']:
          if dep > -1 and mapping[dep]['deps'] is not None:
            template['data-deps'].extend ([d for d in mapping[dep]['deps'] if d not in template['data-deps']])
        ishape = shapes[layer['inputs'][0]]
        batch = ishape[0]
        embdim = ishape[1]
        nhead = shapes[layer['inputs'][12]][0]
        headdim = int (embdim / nhead)
        
        # QKV
        qkvdeps = list ()
        node = {
          'id': len (workload),
          'name': template['name'] + f'_q',
          'data-deps': [dep for dep in template['data-deps']],
          'duration': 1,
          'type': COMP_NODE,
          'num-ops': 2 * batch * embdim * nhead * headdim,
          'size': batch * embdim + embdim * nhead * headdim + batch * nhead * headdim 
        }
        node['size'] = node['size'] * 2
        qkvdeps.append (len (workload))
        workload.append (node)
        node = {
          'id': len (workload),
          'name': template['name'] + f'_k',
          'data-deps': [dep for dep in template['data-deps']],
          'duration': 1,
          'type': COMP_NODE,
          'num-ops': 2 * batch * embdim * nhead * headdim,
          'size': batch * embdim + embdim * nhead * headdim + batch * nhead * headdim 
        }
        node['size'] = node['size'] * 2
        qkvdeps.append (len (workload))
        workload.append (node)
        node = {
          'id': len (workload),
          'name': template['name'] + f'_v',
          'data-deps': [dep for dep in template['data-deps']],
          'duration': 1,
          'type': COMP_NODE,
          'num-ops': 2 * batch * embdim * nhead * headdim,
          'size': batch * embdim + embdim * nhead * headdim + batch * nhead * headdim 
        }
        node['size'] = node['size'] * 2
        qkvdeps.append (len (workload))
        workload.append (node)
        # scores
        scoredeps = list ()
        node = {
          'id': len (workload),
          'name': template['name'] + f'_attention_scores',
          'data-deps': [dep for dep in qkvdeps],
          'duration': 1,
          'type': COMP_NODE,
          'num-ops': 2 * batch * nhead * headdim * batch * sequence,
          'size': batch * nhead + batch * sequence * nhead * headdim + batch * batch * sequence
        }
        node['size'] = node['size'] * 2
        scoredeps.append (len (workload))
        workload.append (node)
        # softmax
        softdeps = list ()
        node = {
          'id': len (workload),
          'name': template['name'] + f'_attention_softmax',
          'data-deps': [dep for dep in scoredeps],
          'duration': 1,
          'type': COMP_NODE,
          'num-ops': 4 * batch * batch * sequence,
          'size': batch * batch * sequence
        }
        node['size'] = node['size'] * 2
        softdeps.append (len (workload))
        workload.append (node)
        # output
        outputdeps = list ()
        node = {
          'id': len (workload),
          'name': template['name'] + f'_attention_output',
          'data-deps': [dep for dep in softdeps],
          'duration': 1,
          'type': COMP_NODE,
          'num-ops': 2 * batch * batch * sequence * nhead * headdim,
          'size': batch * batch * sequence + batch * sequence * nhead * headdim + batch * nhead * headdim
        }
        node['size'] = node['size'] * 2
        outputdeps.append (len (workload))
        workload.append (node)
        # proj
        node = {
          'id': len (workload),
          'name': template['name'] + f'_attention_projection',
          'data-deps': [dep for dep in outputdeps],
          'duration': 1,
          'type': COMP_NODE,
          'num-ops': 2 * batch * nhead * headdim * embdim,
          'size': batch * nhead * headdim + nhead * headdim * embdim + batch * embdim
        }
        node['size'] = node['size'] * 2
        workload.append (node)
        mapping[i]['id'] = node['id']
        shapes[layer['output']] = tuple ([batch, embdim])
      else:
        print ('unknown plugin given:', layer['id'])
        sys.exit (1)
    elif layer['type'] == 'Sigmoid': # elementwise sigmoid
      mapping[i] = dict ()
      mapping[i]['deps'] = None
      node = {
        'id': len (workload),
        'name': layer['id'],
        'data-deps': [
          mapping[dep]['id'] for dep in layer['dependencies'] if dep > -1 and mapping[dep]['id'] > -1
        ],
        'duration': 1, # just a placeholder value
        'type': COMP_NODE,
      }
      # add non compute layer dependencies
      for dep in layer['dependencies']:
        if dep > -1 and mapping[dep]['deps'] is not None:
          node['data-deps'].extend ([d for d in mapping[dep]['deps'] if d not in node['data-deps']])
      node['num-ops'] = int (4 * np.prod (shapes[layer['inputs'][0]]))
      node['size'] = int (np.prod (shapes[layer['inputs'][0]]))
      mapping[i]['id'] = node['id']
      shapes[layer['output']] = shapes[layer['inputs'][0]]
      workload.append (node)
    elif layer['type'] == 'Sub': # elementwise subtraction
      mapping[i] = dict ()
      mapping[i]['deps'] = None
      node = {
        'id': len (workload),
        'name': layer['id'],
        'data-deps': [
          mapping[dep]['id'] for dep in layer['dependencies'] if dep > -1 and mapping[dep]['id'] > -1
        ],
        'duration': 1, # just a placeholder value
        'type': COMP_NODE,
      }
      # add non compute layer dependencies
      for dep in layer['dependencies']:
        if dep > -1 and mapping[dep]['deps'] is not None:
          node['data-deps'].extend ([d for d in mapping[dep]['deps'] if d not in node['data-deps']])
      node['num-ops'] = int (np.prod (shapes[layer['inputs'][0]]))
      node['size'] = 2 * node['num-ops']
      mapping[i]['id'] = node['id']
      shapes[layer['output']] = shapes[layer['inputs'][0]]
      workload.append (node)
    # data movement
    elif layer['type'] == 'Constant': # weight matrix
      mapping[i] = dict ()
      mapping[i]['deps'] = None
      mapping[i]['id'] = -1
      deps = list ()
      for dep in layer['dependencies']:
        if dep > -1 and mapping[dep]['id'] == -1 and mapping[dep]['deps'] is not None:
          deps.extend ([d for d in mapping[dep]['deps'] if d not in deps])
        elif dep > -1 and mapping[dep]['id'] > -1:
          deps.append (mapping[dep]['id'])
      if len (deps) > 0:
        mapping[i]['deps'] = deps
      else:
        mapping[i]['deps'] = None
      shapes[layer['output']] = tuple ([int (s) for s in layer['details']['shape']])
    elif layer['type'] == 'Gather': # retrieve
      mapping[i] = dict ()
      mapping[i]['deps'] = None
      mapping[i]['id'] = -1
      deps = list ()
      for dep in layer['dependencies']:
        if dep > -1 and mapping[dep]['id'] == -1 and mapping[dep]['deps'] is not None:
          deps.extend ([d for d in mapping[dep]['deps'] if d not in deps])
        elif dep > -1 and mapping[dep]['id'] > -1:
          deps.append (mapping[dep]['id'])
      if len (deps) > 0:
        mapping[i]['deps'] = deps
      else:
        mapping[i]['deps'] = None
      axis = layer['details']['axis']
      oshape = tuple ([s if j != axis else shapes[layer['inputs'][1]][0] for j, s in enumerate (shapes[layer['inputs'][0]])])
      shapes[layer['output']] = oshape 
    # data transformation
    elif layer['type'] == 'Shape': # returns tensor shape
      mapping[i] = dict ()
      mapping[i]['deps'] = None
      mapping[i]['id'] = -1
      deps = list ()
      for dep in layer['dependencies']:
        if dep > -1 and mapping[dep]['id'] == -1 and mapping[dep]['deps'] is not None:
          deps.extend ([d for d in mapping[dep]['deps'] if d not in deps])
        elif dep > -1 and mapping[dep]['id'] > -1:
          deps.append (mapping[dep]['id'])
      if len (deps) > 0:
        mapping[i]['deps'] = deps
      else:
        mapping[i]['deps'] = None
      shapes[layer['output']] = tuple ([len (shapes[layer['inputs'][0]])])
    elif layer['type'] == 'Concat': # concatenates two tensors
      mapping[i] = dict ()
      mapping[i]['deps'] = None
      mapping[i]['id'] = -1
      deps = list ()
      for dep in layer['dependencies']:
        if dep > -1 and mapping[dep]['id'] == -1 and mapping[dep]['deps'] is not None:
          deps.extend ([d for d in mapping[dep]['deps'] if d not in deps])
        elif dep > -1 and mapping[dep]['id'] > -1:
          deps.append (mapping[dep]['id'])
      if len (deps) > 0:
        mapping[i]['deps'] = deps
      else:
        mapping[i]['deps'] = None
      axis = layer['details']['axis']
      try:
        newshape = sum ([shapes[layer['inputs'][i]][axis] for i in range (layer['details']['num_inputs'])])
        oshape = tuple ([s if i != axis else newshape for i, s in enumerate (shapes[layer['inputs'][0]])])
      except:
        print (shapes[layer['inputs'][0]])
        sys.exit (1)
      shapes[layer['output']] = oshape 
    elif layer['type'] == 'Reshape': # changes the tensor shape
      mapping[i] = dict ()
      mapping[i]['deps'] = None
      mapping[i]['id'] = -1
      deps = list ()
      for dep in layer['dependencies']:
        if dep > -1 and mapping[dep]['id'] == -1 and mapping[dep]['deps'] is not None:
          deps.extend ([d for d in mapping[dep]['deps'] if d not in deps])
        elif dep > -1 and mapping[dep]['id'] > -1:
          deps.append (mapping[dep]['id'])
      if len (deps) > 0:
        mapping[i]['deps'] = deps
      else:
        mapping[i]['deps'] = None
      shapes[layer['output']] = shapes[layer['inputs'][0]]
    elif layer['type'] == 'Cast': # changes the tensor shape
      mapping[i] = dict ()
      mapping[i]['deps'] = None
      mapping[i]['id'] = -1
      deps = list ()
      for dep in layer['dependencies']:
        if dep > -1 and mapping[dep]['id'] == -1 and mapping[dep]['deps'] is not None:
          deps.extend ([d for d in mapping[dep]['deps'] if d not in deps])
        elif dep > -1 and mapping[dep]['id'] > -1:
          deps.append (mapping[dep]['id'])
      if len (deps) > 0:
        mapping[i]['deps'] = deps
      else:
        mapping[i]['deps'] = None
      shapes[layer['output']] = shapes[layer['inputs'][0]]
    elif layer['type'] == 'Slice': # extracts a subtensor
      mapping[i] = dict ()
      mapping[i]['deps'] = None
      mapping[i]['id'] = -1
      deps = list ()
      for dep in layer['dependencies']:
        if dep > -1 and mapping[dep]['id'] == -1 and mapping[dep]['deps'] is not None:
          deps.extend ([d for d in mapping[dep]['deps'] if d not in deps])
        elif dep > -1 and mapping[dep]['id'] > -1:
          deps.append (mapping[dep]['id'])
      if len (deps) > 0:
        mapping[i]['deps'] = deps
      else:
        mapping[i]['deps'] = None
      shapes[layer['output']] = shapes[layer['inputs'][0]] 
    else:
      print ('unknown layer type:', layer['type'])
      sys.exit (1)
  if verbose and rank == 0:
    for k, v in shapes.items ():
      print (f'{k}\n{v}\n')
  return workload

def main ():
  args = parse_arguments ()
  # get all inputs
  files = os.listdir (args.dir)
  pattern = args.input + r'\.(?P<id>\d+)\.json'
  onnxs = list ()
  for f in files:
    m = re.match (pattern, f)
    if m is None:
      continue
    onnxs.append ((int (m.group ('id')), os.path.join (args.dir, f)))
  n = len (onnxs)
  print (f'found {n} workloads')
  print ('building chakra graph...')
  workload = dict ()
  for i, onnx in onnxs:
    workload[i] = dict ()
    workload[i]['nodes'] = convert (onnx, args.batch, i, n, int (args.split_sends), args.sequence, args.verbose)
  with open (args.output, 'w') as output:
    yaml.dump ({'workload': workload}, output, default_flow_style = False)

if __name__ == '__main__': 
  main ()
