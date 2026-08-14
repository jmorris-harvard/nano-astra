#!/usr/bin/python3

import argparse
import json
import os
import sys

import pandas as pd
import yaml

from astragen import (
  encode,
  GlobalMetadata,
  COMP_NODE,
  COMM_SEND_NODE,
  COMM_RECV_NODE
)
import etgenerate

def parse_arguments ():
  parser = argparse.ArgumentParser (
    prog = 'ASTRA-hf-workload-generator',
    description = 'Builds a chakra execution trace workload from per-kernel '
                  'GPU profiling data and a huggingface model config.',
    epilog = '')
  parser.add_argument ('csv', type = str,
    help = 'CSV file of profiled kernels (id, name, tensor parallelism, batch, sequence, runtime, power)')
  parser.add_argument ('config', type = str,
    help = 'huggingface model config.json file')
  parser.add_argument ('batch', type = int,
    help = 'batch size')
  parser.add_argument ('sequence', type = int,
    help = 'sequence length')
  parser.add_argument ('tensor_parallel', type = int, nargs = '?', default = 1,
    help = 'tensor parallelism degree (default: 1)')
  parser.add_argument ('pipeline_parallel', type = int, nargs = '?', default = 1,
    help = 'pipeline parallelism degree (default: 1)')
  parser.add_argument ('gpu_type', type = str,
    help = 'GPU type the profile was collected on (e.g. A100, H100)')
  parser.add_argument ('-o', '--output', type = str, default = None,
    help = 'output prefix for generated files (default derived from config/gpu_type)')
  parser.add_argument ('-f', '--format', type = str, choices = ['et', 'yaml'], default = 'et',
    help = "output format: 'et' writes one chakra .et per gpu (default); "
           "'yaml' writes a single human-readable .yaml workload for inspection")
  return parser.parse_args ()

def load_config (path):
  with open (path, 'r') as f:
    return json.load (f)

def load_kernels (csv_path, tensor_parallel, batch, sequence):
  df = pd.read_csv (csv_path)
  kernels = []
  for kernel_id in sorted (df['ID'].unique ()):
    match = df[
      (df['ID'] == kernel_id) &
      (df['Tensor Parallelism'] == tensor_parallel) &
      (df['Batch'] == batch) &
      (df['Sequence'] == sequence)
    ]
    if match.empty:
      print (
        'error: no kernel found in %s for id=%d, tensor_parallel=%d, batch=%d, sequence=%d' %
        (csv_path, kernel_id, tensor_parallel, batch, sequence),
        file = sys.stderr)
      sys.exit (1)
    row = match.iloc[0]
    kernels.append ({
      'name': row['Name'],
      'id': int (row['ID']),
      'runtime': float (row['Runtime (Mean) [ns]']),
      'power': float (row['Power [W]']),
    })
  return kernels

def is_ln_f (name):
  return 'ln_f' in name.lower ()

def is_lm_head (name):
  return 'lm_head' in name.lower ()

def is_attention (name):
  return 'gptattention' in name.lower ().replace ('_', '')

def build_workload (kernels, config, batch, sequence, tensor_parallel, pipeline_parallel):
  total_gpus = tensor_parallel * pipeline_parallel
  gpu_nodes = [[] for _ in range (total_gpus)]

  num_hidden_layers = config['num_hidden_layers']
  layers_per_group = num_hidden_layers // pipeline_parallel
  if layers_per_group == 0:
    print (
      'error: num_hidden_layers (%d) is smaller than pipeline_parallel (%d), '
      'each pipeline stage needs at least one layer' %
      (num_hidden_layers, pipeline_parallel),
      file = sys.stderr)
    sys.exit (1)
  # layers that don't divide evenly across pipeline stages are spread one at
  # a time across the first groups, so no group has more than one extra layer
  remainder_layers = num_hidden_layers % pipeline_parallel

  comm_size = batch * sequence * config['hidden_size'] * 2 # 16 bits = 2 bytes
  tag = 0

  def add_node (gpu, node):
    node['id'] = len (gpu_nodes[gpu])
    prev = gpu_nodes[gpu][-1]['id'] if len (gpu_nodes[gpu]) > 0 else None
    node['data-deps'] = [prev] if prev is not None else []
    gpu_nodes[gpu].append (node)

  for group in range (pipeline_parallel):
    group_gpus = list (range (group * tensor_parallel, (group + 1) * tensor_parallel))
    is_last_group = group == pipeline_parallel - 1
    group_layers = layers_per_group + (1 if group < remainder_layers else 0)
    for layer in range (group_layers):
      is_last_layer = layer == group_layers - 1
      for kernel in kernels:
        if is_ln_f (kernel['name']) or is_lm_head (kernel['name']):
          continue
        for gpu in group_gpus:
          add_node (gpu, {
            'type': COMP_NODE,
            'name': kernel['name'],
            'live-runtime': kernel['runtime'],
            'live-power': kernel['power'],
          })
        if tensor_parallel > 1 and is_attention (kernel['name']):
          for src in group_gpus:
            for dst in group_gpus:
              if src == dst:
                continue
              add_node (src, {
                'type': COMM_SEND_NODE,
                'name': kernel['name'] + '_TP_SEND',
                'size': comm_size,
                'tag': tag,
                'src': src,
                'dst': dst,
              })
              add_node (dst, {
                'type': COMM_RECV_NODE,
                'name': kernel['name'] + '_TP_RECV',
                'size': comm_size,
                'tag': tag,
                'src': src,
                'dst': dst,
              })
              tag = tag + 1

      # the final layer of the final pipeline stage also runs the trailing
      # ln_f kernels (they only occur once, after the last transformer layer)
      if is_last_group and is_last_layer:
        for kernel in kernels:
          if not is_ln_f (kernel['name']) and not is_lm_head (kernel['name']):
            continue
          for gpu in group_gpus:
            add_node (gpu, {
              'type': COMP_NODE,
              'name': kernel['name'],
              'live-runtime': kernel['runtime'],
              'live-power': kernel['power'],
            })

      # end of layer: pipeline hand-off to the next stage
      if pipeline_parallel > 1 and group < pipeline_parallel - 1:
        for position, gpu in enumerate (group_gpus):
          next_gpu = (group + 1) * tensor_parallel + position
          add_node (gpu, {
            'type': COMM_SEND_NODE,
            'name': 'PIPELINE_SEND',
            'size': comm_size,
            'tag': tag,
            'src': gpu,
            'dst': next_gpu,
          })
          add_node (next_gpu, {
            'type': COMM_RECV_NODE,
            'name': 'PIPELINE_RECV',
            'size': comm_size,
            'tag': tag,
            'src': gpu,
            'dst': next_gpu,
          })
          tag = tag + 1

  return gpu_nodes

def write_workload_et (gpu_nodes, output_prefix):
  for gpu, nodes in enumerate (gpu_nodes):
    filename = '%s.%d.et' % (output_prefix, gpu)
    with open (filename, 'wb') as et:
      encode (et, GlobalMetadata (version = '0.0.4'))
      etgenerate.generate (gpu, nodes, et)
    print ('wrote %s (%d nodes)' % (filename, len (nodes)))

def write_workload_yaml (gpu_nodes, output_prefix):
  # dump a single human-readable yaml describing every gpu's node list, using
  # the same {'workload': {npu: {'nodes': [...]}}} layout randomize.py emits so
  # it can be eyeballed (and, if desired, fed through the yaml workload path)
  workload = {}
  for gpu, nodes in enumerate (gpu_nodes):
    workload[gpu] = {'nodes': nodes}
  filename = '%s.yaml' % (output_prefix)
  with open (filename, 'w') as yml:
    yaml.dump ({'workload': workload}, yml, default_flow_style = False)
  total = sum (len (nodes) for nodes in gpu_nodes)
  print ('wrote %s (%d gpus, %d nodes total)' % (filename, len (gpu_nodes), total))

def main ():
  args = parse_arguments ()

  config = load_config (args.config)
  kernels = load_kernels (args.csv, args.tensor_parallel, args.batch, args.sequence)
  gpu_nodes = build_workload (
    kernels, config, args.batch, args.sequence,
    args.tensor_parallel, args.pipeline_parallel)

  output_prefix = args.output
  if output_prefix is None:
    model_name = os.path.splitext (os.path.basename (args.config))[0]
    output_prefix = '%s_%s_tp%d_pp%d_b%d_s%d' % (
      model_name, args.gpu_type, args.tensor_parallel,
      args.pipeline_parallel, args.batch, args.sequence)

  if args.format == 'yaml':
    write_workload_yaml (gpu_nodes, output_prefix)
  else:
    write_workload_et (gpu_nodes, output_prefix)

if __name__ == '__main__':
  main ()
