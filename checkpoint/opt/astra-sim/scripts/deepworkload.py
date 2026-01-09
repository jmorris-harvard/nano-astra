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
def generate_workload (npus, output = 'dl.yaml', verbose = False):
  # init output
  workload = {}

  with open (output, 'w') as yml:
    yaml.dump ({'workload': workload}, yml, default_flow_style = False)

def main ():
  parser = argparse.ArgumentParser (
    prog = 'ASTRA-workload-generator',
    description = '',
    epilog = '')
  parser.add_argument ('-n', '--npus', type = int, required = True)
  parser.add_argument ('-o', '--output', type = str, required = True)
  parser.add_argument ('-v', '--verbose', action = 'store_true', default = False)
  parser.add_argument ('--seed', type = int, default = None)
  args = parser.parse_args (sys.argv[1:])
  # assert args
  random.seed (args.seed)
  generate_workload (
    args.npus,
    args.output,
    args.verbose
  )

if __name__ == '__main__':
  main ()
