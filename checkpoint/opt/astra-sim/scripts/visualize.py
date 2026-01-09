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

def build (filename, target_design, verbose = False, placement = 'any'):
  yml = None
  with open (filename, 'r') as file:
    yml = yaml.load (file, Loader = Loader)
  if target_design not in yml.keys ():
    print ('Could not find (%s) in (%s)' % (target_design, filename))
  design = yml[target_design]

  # get links
  links = design['ns3']['topology']['links']

  # get nodes
  nnpus = design['dims'][0]['npus']
  
  # plot
  g = nx.Graph ()
  node_names = []
  for link in links:
    g.add_edge (int (link['node-a']), int (link['node-b']))
    if link['node-a'] not in node_names:
      node_names.append (int (link['node-a']))
    if link['node-b'] not in node_names:
      node_names.append (int (link['node-b']))

  # count nodes
  nswitches = max (node_names) - nnpus + 1

  if placement == 'inline':
    # place npus on bottom and switches on top
    pos = {}
    # npus
    if nnpus % 2 == 0:
      midpoint = nnpus / 2
      # even, leave 0 blank
      for i in range (nnpus):
        pos[i] = (i - midpoint, 0) if i < midpoint else (i - midpoint + 1, 0)
    else:
      midpoint = int (nnpus / 2)
      # odd overlap
      for i in range (nnpus):
        pos[i] = (i - midpoint, 0)
    # switches
    if nswitches % 2 == 0:
      midpoint = nswitches / 2
      # even, leave 0 blank
      for i in range (nswitches):
        pos[i + nnpus] = (i - midpoint, 1) if i < midpoint else (i - midpoint + 1, 1)
    else:
      midpoint = int (nswitches / 2)
      # odd overlap
      for i in range (nswitches):
        pos[i + nnpus] = (i - midpoint, 1)
  elif placement == 'any':
    # let library decide placement
    pos = nx.spring_layout (g)
  elif placement == 'heirarchy':
    heirarchy = design['ns3']['topology']['heirarchy']
    # place npus inline on bottom
    pos = {}
    # npus
    midpoint = int (nnpus / 2)
    if nnpus % 2 == 0:
      # even, leave 0 blank
      for i in range (nnpus):
        pos[i] = (1 * (i - midpoint), 0) if i < midpoint else (1 * (i - midpoint + 0.25), 0)
    else:
      # odd overlap
      for i in range (nnpus):
        pos[i] = (1 * (i - midpoint), 0)
    # switches
    for key in sorted (heirarchy.keys ()):
      nsw = len (heirarchy[key])
      midpoint = int (nsw / 2)
      for i, switch in enumerate (heirarchy[key]):
        if nsw % 2 == 0:
          pos[switch] = (1 * (i - midpoint), int (key)) if i < midpoint else (1 * (i - midpoint + 0.25), int (key))
        else:
          pos[switch] = (1 * (i - midpoint), int (key))
  else:
    print ('placement strategy (%s) not defined' % (placement))
    sys.exit ()

  return g, pos, nnpus

class Animation ():
  def __init__ (this, nnpus, g, pos):
    fig, ax = plt.subplots (figsize=(15,8))
    this.fig = fig
    this.ax = ax
    this.nnpus = nnpus
    this.g = g
    this.pos = pos

    # base colors
    this.ncolor = 'orange'
    this.scolor = 'blue'
    this.fcolor = 'white'
    this.inactcolor = 'black'
    this.actcolor = 'red'

    # other settings
    this.linewidth = 1
    this.nodesize = 1000

  def update (this, num):
    # clear figure
    this.ax.clear ()

    # add node colors
    ncolors = [this.ncolor if node < this.nnpus else this.scolor for node in this.g.nodes ()]
    # add node activity colors
    bcolors = [this.actcolor if str (node) in this.frames[num] else this.inactcolor for node in this.g.nodes ()]

    # add edge activity colors
    ecolors = []
    for u,v in this.g.edges ():
      u, v = min (u, v), max (u, v)
      estr = f'{u}_{v}'
      if estr in this.frames[num]:
        ecolors.append (this.actcolor)
      else:
        ecolors.append (this.inactcolor)

    nx.draw_networkx_nodes (
      this.g,
      this.pos,
      ax = this.ax,
      nodelist = this.g.nodes (),
      node_size = this.nodesize,
      node_color = ncolors,
      edgecolors = bcolors,
      linewidths = this.linewidth)

    nx.draw_networkx_labels (
      this.g,
      this.pos,
      ax = this.ax,
      font_color = this.fcolor)

    nx.draw_networkx_edges (
      this.g,
      this.pos,
      ax = this.ax,
      edgelist = this.g.edges (),
      edge_color = ecolors,
      width = this.linewidth)

    # add legend
    handles = [
      mpatches.Patch (color=this.ncolor, label='Compute Node'),
      mpatches.Patch (color=this.scolor, label='Switch Node'),
      mpatches.Patch (color=this.inactcolor, label='Inactive Link'),
      mpatches.Patch (color=this.actcolor, label='Active Link'),
    ]
    this.ax.legend (handles=handles, loc='upper right')

  def draw (this, output, filename = None):
    # create animation pass list
    frames = [[]]
    if filename is not None:
      with open (filename) as ffile:
        lines = ffile.read ().splitlines ()
        active = [line.split (',') for line in lines]
        frames.extend (active)
    this.frames = frames

    # save first frame
    this.update (0)
    plt.savefig (output)

    # do animation
    if filename is not None: 
      def update (num):
        this.update (num)
      ani = animation.FuncAnimation (
        this.fig,
        update,
        frames = len (this.frames),
        repeat = True)
      # show animation
      # ani.save (filename = 'animation.gif')
      plt.show ()

def main ():
  parser = argparse.ArgumentParser (
    prog = 'ASTRA-topology-visualizer',
    description = '',
    epilog = '')
  parser.add_argument ('-c', '--configuration', type = str, required = True)
  parser.add_argument ('-d', '--design', type = str, required = True)
  parser.add_argument ('-o', '--output', type = str, required = True)
  parser.add_argument ('-v', '--verbose', action = 'store_true', default = False)
  parser.add_argument ('-p', '--placement', default = 'any')
  parser.add_argument ('-a', '--animate', default = None)
  args = parser.parse_args (sys.argv[1:])
  g, pos, nnpus = build (args.configuration, args.design, args.verbose, args.placement)
  anime = Animation (nnpus, g, pos)
  anime.draw (args.output, args.animate)

if __name__ == '__main__':
  main ()
