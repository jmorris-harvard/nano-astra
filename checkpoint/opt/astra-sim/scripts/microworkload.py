#!/usr/bin/python3

import argparse
import numpy as np
import os
import pandas as pd
import random
import sys
import yaml

from microbench import (
  all2all,
  allgather,
  allreduce,
  mlp
)

def main (benchmark, nnpus, output, size, batch, layers):
  workload = None
  if benchmark.lower () == 'mlp'.lower ():
    workload, _ = mlp (nnpus, size, batch, layers)
  elif benchmark.lower () == 'all2all'.lower ():
    workload, _ = all2all (nnpus, size)
  elif benchmark.lower () == 'allreduce'.lower ():
    workload, _ = allreduce (nnpus, size)
  elif benchmark.lower () == 'allgather'.lower ():
    workload, _ = allgather (nnpus, size)
  else:
    print ('invalid benchmark requested, %s' % (benchmark))
    sys.exit ()

  with open (output, 'w') as yml:
    yaml.dump ({'workload': workload}, yml, default_flow_style = False)

if __name__ == '__main__': 
  parser = argparse.ArgumentParser (
    prog = 'ASTRA-microbenchmark-generator',
    description = '',
    epilog = '')
  parser.add_argument ('-b', '--benchmark', type = str, required = True)
  parser.add_argument ('-n', '--npus', type = int, required = True)
  parser.add_argument ('-o', '--output', type = str, required = True)
  parser.add_argument ('-s', '--size', type = int, default = 1024)
  parser.add_argument ('-e', '--batch', type = int, default = 1024)
  parser.add_argument ('-l', '--layers', type = int, default = 1024)
  args = parser.parse_args ()
  main (args.benchmark, args.npus, args.output, args.size, args.batch, args.layers)
