#!/usr/bin/python3

import argparse
import os
import random
import sys
import yaml

import etgenerate
from astragen import (
  GlobalMetadata,
  encode,
  COMP_NODE
)

def main ():
  parser = argparse.ArgumentParser (
    prog = 'ASTRA-dummy-workload-generator',
    description = 'creates empty workload (0 operation compute node) to ensure all compute elements have jobs',
    epilog = '')
  parser.add_argument ('-b',
                       '--basename',
                       type = str,
                       required = True,
                       help = 'execution trace filename prefix')
  parser.add_argument ('-n',
                       '--nodes',
                       nargs = '+',
                       default = [],
                       help = 'node number list that corresponds to nodes that need empty workloads')
  args = parser.parse_args (sys.argv[1:])
  workload = dict ()
  for npu in args.nodes:
    workload[int (npu)] = dict ()
    workload[int (npu)]['nodes'] = [
        {
          'id': 0,
          'name': 'DUMMY',
          'type': COMP_NODE,
          'data-deps': [],
          'num-ops': 0,
          'size': 1
        }
    ]
    with open (args.basename + '.' + str (npu) + '.et', 'wb') as et:
      encode (et, GlobalMetadata (version = '0.0.4'))
      etgenerate.generate (npu, workload[int (npu)]['nodes'])

if __name__ == '__main__':
  main ()
