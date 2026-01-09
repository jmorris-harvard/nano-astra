#!/usr/bin/python3

import argparse
import numpy as np
import os
import matplotlib
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import random
import sys
import yaml
try:
  from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
  from yaml import Loader, Dumper

def build (nodes, filename):
  # map ids to names
  mapping = dict ()
  dependencies = list ()
  unfinished = list ()
  for node in nodes:
    # add node to mapping
    mapping[int (node['id'])] = node['name']
    # add dependencies to graph
    for dep in node['data-deps']:
      if int (dep) not in mapping.keys ():
        unfinished.append (
          (node['name'], int (dep))
        )
      else:
        dependencies.append (
          (node['name'], mapping[int (dep)])
        )
  # add all unfinished
  for name, dep in unfinished:
    dependencies.append (
      (name, mapping[dep])
    )

  g = nx.DiGraph()
  g.add_edges_from(dependencies)
  pos = nx.spring_layout(g, seed=0, k = 2.75)

  fig, ax = plt.subplots(figsize = (10,10))
  nx.draw_networkx_edges(g, pos, ax=ax, arrowstyle='->', arrowsize=15, edge_color='black')
  max_text_length = max(len(node) for node in g.nodes)
  rectangle_width = max_text_length * 0.025
  rectangle_height = 0.05

  for node, (x, y) in pos.items():
    rect = mpatches.Rectangle((x - rectangle_width / 2, y - rectangle_height / 2), rectangle_width, rectangle_height, color='lightblue', ec='black')
    ax.add_patch(rect)
    ax.text(x, y, node, horizontalalignment='center', verticalalignment='center', fontsize=8, fontweight='bold')

  ax.axis('off')
  ax.set_aspect('equal')
  ax.set_title(os.path.splitext(os.path.split (filename)[1])[0], fontsize=12)
  fig.savefig (filename)

def main ():
  parser = argparse.ArgumentParser (
    prog = 'ASTRA-workload-visualizer',
    description = '',
    epilog = '')
  parser.add_argument ('-f', '--filename', type = str, required = True)
  args = parser.parse_args (sys.argv[1:])
  yml = None
  with open (args.filename, 'r') as ymlFile:
    yml = yaml.load (ymlFile, Loader = Loader)
  workload = yml['workload']
  for worker in workload.keys ():
    build (workload[worker]['nodes'], os.path.splitext (args.filename)[0] + '_' + str (worker) + '.png')

if __name__ == '__main__':
  main ()
