#!/usr/bin/python3

import argparse
import numpy as np
import os
import pandas as pd
import random
import sys
import yaml

from enum import Enum

class Node:
  ID = 0

  class Type (Enum):
    UNKNOWN = 0
    COMP = 1
    SWITCH = 2
    LINK = 3
    CONTAINER = 4
  
  def __init__ (this):
    this._ports = []
    # assign id and increment
    this._id = Node.ID
    Node.ID = Node.ID + 1
    # add type and description
    this._type = Node.Type.UNKNOWN
    # add power consumption
    this._idle = 0
    this._peak = 0

  def idle (this, nidle = None):
    if nidle != None:
      this._idle = nidle
    return this._idle

  def peak (this, npeak = None):
    if npeak != None:
      this._peak = npeak
    return this._peak

  def ports (this):
    return this._ports

  def n (this):
    return len (this.ports ())

  def type (this):
    return this._type

  def id (this):
    return this._id

  def add (this, node):
    if (this.type () == Node.Type.COMP or this.type () == Node.Type.SWITCH) and \
        node.type () != Node.Type.LINK:
      print ('must link to comp/switch node')
      sys.exit ()
    elif this.type () == Node.Type.LINK and node.type () == Node.Type.LINK:
      print ('must add comp/switch to link node')
      sys.exit ()
    else:
      this.ports ().append (node)

  def rem (this, id):
    i = None
    for j, node in enumerate (this.ports ()):
      if node.id () == id:
        i = j
        break
    if i is None:
      print ('node (%s) does not exist in ports node (%s)' % (id, this.id ()))
      sys.exit ()
    return this.ports ().pop (i)

  def peek (this, id):
    for node in this.ports ():
      if node.id () == id:
        return node

  def flatten (this, searched = None):
    # initialize
    if not searched:
      searched = {
        'comp': [],
        'switch': [],
        'link': [],
      }
    # check id in appropriate subsection
    exists = False
    if this.type () == Node.Type.COMP or \
       this.type () == Node.Type.SWITCH or \
       this.type () == Node.Type.LINK:
      bucket = None
      if this.type () == Node.Type.COMP:
        bucket = searched['comp']
      elif this.type () == Node.Type.SWITCH:
        bucket = searched['switch']
      elif this.type () == Node.Type.LINK:
        bucket = searched['link']
      # search bucket
      for node in bucket:
        if node.id () == this.id ():
          exists = True
 
      if not exists:
        # add to bucket
        bucket.append (this)
    if this.type == Node.Type.CONTAINER or not exists:
      # pass to ports
      for node in this.ports ():
        searched = node.flatten (searched)
    return searched

  def __str__ (this):
    ids = [str(node.id()) for node in this.ports ()]
    typestr = ''
    if this._type == Node.Type.COMP:
      typestr = 'COMP'
    elif this._type == Node.Type.SWITCH:
      typestr = 'SWITCH'
    elif this._type == Node.Type.LINK:
      typestr = 'LINK'
    out = '<%s:%s (%s)>' % (typestr, this.id (), ' '.join (ids))
    return out

class GenericLink (Node):
  def __init__ (this, src, dst = None):
    super ().__init__ ()
    this._type = Node.Type.LINK
    # set connections
    this.add (src)
    src.add (this)
    if dst is not None:
      dst.add (this)
      this.add (dst)
    # set specs
    this._bw = 1e9
    this._latency = 500e-9

  def bw (this, nbw = None):
    if nbw is not None:
      this._bw = nbw
    return this._bw

  def latency (this, nlatency = None):
    if nlatency is not None:
      this._latency = nlatency
    return this._latency

  def anode (this):
    if len (this._ports) > 0:
      return this._ports[0]

  def bnode (this):
    if len (this._ports) > 1:
      return this._ports[1]

class NVLink (GenericLink):
  def __init__ (this, src, dst = None):
    super ().__init__ (src, dst)
    # add specs
    # using NVLink Ampere 2nd gen as bandwidth and latency
    # https://www.nvidia.com/en-us/data-center/nvlink/
    # https://community.fs.com/article/an-overview-of-nvidia-nvlink.html
    # 1.3pJ per bit
    this.bw (800e9) # 100GBps
    this.latency (434e-9) # switch latency
    # this.latency (50e-9) # switch latency
    this.idle (0.0)
    this.peak (5.5 * 2) # change this

class Infiniband (GenericLink):
  def __init__ (this, src, dst = None):
    super ().__init__ (src, dst)
    # add specs
    # using QM8790 switch as bandwidth
    # 130ns Port to Port Latency
    # https://network.nvidia.com/files/doc-2020/pb-qm8790.pdf
    # https://docs.nvidia.com/networking/display/qm87xx/specifications
    # requires login for power consumption
    this.bw (200e9) # 200Gbps
    # using ??? as latency
    # using QSFP56 cable for latency
    # https://network.nvidia.com/pdf/prod_cables/PB_MFS1S00-HxxxE_200Gbps_QSFP56_AOC.pdf
    # https://docs.nvidia.com/networking/display/mfs1s00hxxxv10
    # 8.7W power consumption
    # 10W power consumption
    this.latency (120e-9 + 600e-9) # switch + cable latency (~5ns per meter) assume 100m
    # this.latency (50e-9)
    this.idle (0.0)
    this.peak (10.0)

class GenericEndpoint (Node):
  def __init__ (this, nports = 1, types = None):
    super ().__init__ ()
    # init ports
    for i in range (nports):
      if types is not None:
        # add each specified link type
        types[i] (this)
      else:
        # add generic link
        GenericLink (this)

  def getlinktype (this, type = GenericLink):
    # get link id of requested type if available
    for link in this.ports ():
      if link.n () == 1 and isinstance (link, type):
        return link.id (), link.anode ()

  def getlinknode (this, id):
    for link in this.ports ():
      n1, n2 = link.anode (), link.bnode ()
      if not n2:
        continue
      if n1.id () == id or n2.id () == id:
        return link.id ()

  def getconnectedlinktype (this, type = GenericLink):
    for link in this.ports ():
      if link.n () == 2 and isinstance (link, type):
        return link.id (), link.anode (), link.bnode ()


  def link (this, node, type = GenericLink):
    # replace link with linked
    (l1, anode), (l2, bnode) = this.getlinktype (type), node.getlinktype (type)
    if l1 is None or l2 is None:
      print ('no available links')
      sys.exit ()
    # remove old links
    this.rem (l1)
    node.rem (l2)
    # add new link
    l3 = type (anode, bnode)

  def unlink (this, node):
    l1 = this.getlinknode (node.id ())
    linktype = type (this.peek (l1))
    # remove old links
    this.rem (l1)
    node.rem (l1)
    # replace with new links
    linktype (this)
    linktype (node)

class GenericSwitch (GenericEndpoint):
  def __init__ (this, nports = 2, types = None):
    super ().__init__ (nports, types)
    this._type = Node.Type.SWITCH

class NVLinkSwitch (GenericSwitch):
  def __init__ (this):
    super ().__init__ (8, [
      NVLink,
      NVLink,
      NVLink,
      NVLink,
      NVLink,
      NVLink,
      NVLink,
      NVLink
    ])

class InfinibandSwitch (GenericSwitch):
  def __init__ (this, nports):
    types = [Infiniband for _ in range (nports)]
    super ().__init__ (nports, types)

class GenericCompute (GenericEndpoint):
  def __init__ (this, nports = 1, types = None):
    super(). __init__ (nports, types)
    this._type = Node.Type.COMP
    # set specs
    this._perf = 1e9
    this._bw = 1e9

  def bw (this, nbw = None):
    if nbw is not None:
      this._bw = nbw
    return this._bw

  def perf (this, nperf = None):
    if nperf is not None:
      this._perf = nperf
    return this._perf

class A100 (GenericCompute):
  def __init__ (this):
    # A100s have 12 NVLINK ports and 1 PCIe Port >> Infiniband Port w/ Adapter
    super ().__init__ (7, [
      NVLink,
      NVLink,
      NVLink,
      NVLink,
      NVLink,
      NVLink,
      Infiniband
    ])
    this.bw (1555.0e9)
    this.perf (156.0e12)
    this.idle (50.0)
    this.peak (400.0)

class DGXA100Unit (GenericCompute):
  def __init__ (this):
    super ().__init__ (2, [Infiniband, Infiniband])
    this.bw (12440.0e9)
    this.perf (1248.0e12)
    this.idle (400.0)
    this.peak (3200.0)

class DGXA100 (GenericEndpoint):
  def __init__ (this):
    # start with 0 links and add on
    super ().__init__ (0)
    this._type = Node.Type.CONTAINER
    # make eight A100 nodes
    this._nodes = [A100 () for _ in range (8)]
    # make six NVLINK switches
    this._int_switches = [NVLinkSwitch () for _ in range (6)]
    # connect GPUs
    for a100 in this._nodes:
      for nvswitch in this._int_switches:
        a100.link (nvswitch, NVLink)
    # create infiniband switch
    this._out_switches = [InfinibandSwitch (16)]
    # link infiniband switches
    for a100 in this._nodes:
      a100.link (this._out_switches[0], Infiniband)
    # grab infiniband links and add on
    for link in this._out_switches[0].ports ():
      if link.n () == 1:
        this.add (link)

class SUA100 (GenericEndpoint):
  def __init__ (this):
    # start with 0 links and add on
    super ().__init__ (0)
    this._type = Node.Type.CONTAINER
    # make 20 DGX units
    this._nodes = [DGXA100 () for _ in range (20)]
    # make 8 leaf switches
    this._switches = [InfinibandSwitch (40) for _ in range (8)]
    for dgx in this._nodes:
      for infiniband in this._switches:
        dgx.link (infiniband, Infiniband)
    # grab infiniband links and add on
    for switch in this._switches:
      for link in switch.ports ():
        if link.n () == 1:
          this.add (link)

class SpineGroupA100 (GenericEndpoint):
  def __init__ (this):
    # start with 0 links and add on
    super ().__init__ (0)
    this._type = Node.Type.CONTAINER
    # make 7 SU
    this._nodes = [SUA100 () for _ in range (7)]
    # make 8 by 10 spine switches
    # each spine switch is connected to a leaf switch corresponding
    #   to its spine group id (7 underconnects)
    # each spine switch is connected to an upper layer 27 switches
    #   corresponding to odd or even core group (27 overconnects)
    this._switches = [[InfinibandSwitch (40) for i in range (10)] for j in range (8)]
    for su in this._nodes:
      # grab switches and connect
      for i, infiniband in enumerate (su._switches):
        # connect leaf switches to all switches in spine group
        for spine in this._switches[i]:
          # connect to spine
          infiniband.link (spine, Infiniband)
    # grab infiniband links and add on
    for sg in this._switches:
      for spine in sg:
        for link in spine.ports ():
          if link.n () == 1:
            this.add (link)

class SuperPodA100 (GenericEndpoint):
  def __init__ (this):
    # start with 0 links and add on
    super ().__init__ (0)
    this._type = Node.Type.CONTAINER
    # make spine group
    this._nodes = [SpineGroupA100 ()]
    # make 2 by 27 core switches
    # each has a link for each spine group switch
    # plus 1 for reachability
    this._switches = [[InfinibandSwitch (40 + 1) for i in range (27)] for j in range (2)]
    for sg in this._nodes[0]._switches:
      # enumerate each switch in each group
      for i, spine in enumerate (sg):
        # link with all switches in
        for core in this._switches[i % 2]:
          core.link (spine, Infiniband)
    # grab infiniband links and add on
    for cg in this._switches:
      for core in cg:
        for link in core.ports ():
          if link.n () == 1:
            this.add (link)

class SuperSpineA100 (GenericEndpoint):
  def __init__ (this):
    # start with 0 links and add on
    super ().__init__ (0)
    this._type = Node.Type.CONTAINER
    # make 4 SU 
    this._nodes = [SUA100 () for _ in range (4)]
    # make 1 by 20 spine switches
    # each spine switch is connected to each leaf node
    # 32 leaf switches (network not fully utilized)
    this._switches = [InfinibandSwitch (40) for _ in range (20)]
    for su in this._nodes:
      # grab switches and connect
      for i, infiniband in enumerate (su._switches):
        # connect leaf switches to all switches in spine group
        for s, spine in enumerate (this._switches):
          infiniband.link (spine, Infiniband)
    # grab infiniband links and add on
    for spine in this._switches:
      for link in spine.ports ():
        if link.n () == 1:
          this.add (link)

class FatTree (GenericEndpoint):
  def __init__ (this, k, N, ComputeType = DGXA100Unit, SwitchType = InfinibandSwitch, LinkType = Infiniband):
    # parse args
    if k % 2 != 0:
      print ('k-ary Fat Tree must be divisible by 2')
      sys.exit ()
    if N % k != 0:
      print ('N ports must be divisible by k')
    # begin with 0 external links
    super ().__init__ (0)
    this._type = Node.Type.CONTAINER
    # get component amounts
    kCompute = int ((k ** 3) / 4)
    kCore = int ((k / 2) ** 2)
    kAgg = int ((k ** 2) / 2)
    kEdge = kAgg
    kMult = int (N / k)
    # compute nodes
    this._nodes = [ComputeType () for _ in range (kCompute)]
    this._switches = [SwitchType (N + 1) for _ in range (kCore)]
    this.agg = [SwitchType (N + 1) for _ in range (kAgg)]
    this.edge = [SwitchType (N + 1) for _ in range (kEdge)]

    # connect core and agg
    cnt = 0
    for i, agg in enumerate (this.agg):
      for j in range (int (k/2)):
        for _ in range (kMult):
          agg.link (this._switches[(i % int (k / 2)) * int (k / 2) + j], LinkType)
          cnt = cnt + 1
    # print (cnt)

    # connect agg and edge
    cnt = 0
    for i in range (k):
      for j in range (int (k/2)):
        for l in range (int (k/2)):
          for _ in range (kMult):
            this.agg[int (i * k / 2) + j].link (this.edge[int (i * k / 2) + l], LinkType)
            cnt = cnt + 1
    # print (cnt)

    # connect edge and compute (kMult removed)
    cnt = 0
    for i, edge in enumerate (this.edge):
      for j in range (int (k / 2)):
        edge.link (this._nodes[int (i * k / 2) + j], LinkType)
        cnt = cnt +1
    # print (cnt)

    for switch in this._switches:
      for link in switch.ports ():
        if link.n () == 1:
          this.add (link)

class LeafSpine (GenericEndpoint):
  def __init__ (this, k, N, ComputeType = DGXA100Unit, SwitchType = InfinibandSwitch, LinkType = Infiniband):
    # parse args
    if k % 2 != 0:
      print ('k-ary Fat Tree must be divisible by 2')
      sys.exit ()
    if N % k != 0:
      print ('N ports must be divisible by k')
      sys.exit ()
    # begin with 0 external links
    super ().__init__ (0)
    this._type = Node.Type.CONTAINER
    # get component amounts
    kCompute = int ((k ** 2) / 2)
    kLeaf = int (k)
    kSpine = int (k / 2)
    kMult = int (N / k)
    # compute nodes
    this._nodes = [ComputeType () for _ in range (kCompute)]
    this._switches = [SwitchType (N + 1) for _ in range (kSpine)]
    this.leaf = [SwitchType (N + 1) for _ in range (kLeaf)]

    # connect spine and leaf
    cnt = 0
    for i, spine in enumerate (this._switches):
      for j in range (kLeaf):
        for _ in range (kMult):
          spine.link (this.leaf[j], LinkType)
          cnt = cnt + 1
    # print (cnt)

    # connect leaf and compute
    cnt = 0
    for i, compute in enumerate (this._nodes):
      compute.link (this.leaf[int (i / (k / 2))], LinkType)
      cnt = cnt +1
    # print (cnt)

    for switch in this._switches:
      for link in switch.ports ():
        if link.n () == 1:
          this.add (link)

class JellyFish (GenericEndpoint):
  def __init__ (this, k, r, N, P, ComputeType = DGXA100Unit, SwitchType = InfinibandSwitch, LinkType = Infiniband):
    # parse args
    if P % k != 0:
      print ('P ports must be divisible by k')
      sys.exit ()
    if k - r < 1:
      print ('Racks must be able to support at least 1 server, k - r at least 1')
      sys.exit ()
    super ().__init__ (0)
    this._type = Node.Type.CONTAINER
    kCompute = int (N * (k - r))
    kToR = int (N)
    kMult = int (P / k)
    this._nodes = [ComputeType () for _ in range (kCompute)]
    this._switches = [SwitchType (P + 1) for _ in range (kToR)]

    # connect ToR to compute
    for i, compute in enumerate (this._nodes):
      print (i, int (i / (k-r)))
      compute.link (this._switches[int (i / (k - r))], LinkType)

    # connect ToR
    cur = []
    nxt = []
    fin = []
    for switch in this._switches:
      cur.append (switch)

    # algorithm (for all rack connections)
    for con in range (r):
      edges = []
      while len (cur) != 0:
        # chose a random node
        na = random.choice (cur)
        ina = cur.index (na)
        cur.pop (ina)
        nb = None
        if len (cur) == 0:
          if con == r - 1:
            # odd one out
            break
          # use node in nxt
          unchecked = [z for z in nxt if z.id () != na.id ()]
          lz = None
          while len (unchecked) != 0:
            nb = random.choice (unchecked)
            if nb.getlinknode (na.id ()) is None:
              # get connection that uses this node
              for n1, n2 in edges:
                if n1.id () == nb.id () or n2.id () == nb.id ():
                  nb = n1
                  lz = n2
              break
            else:
              inb = unchecked.index (nb)
              unchecked.pop (inb)
              nb = None
          if nb is None:
            # all nodes have this node as a neighbor so just grab a random one (in edges)
            nb, lz = random.choice (edges)
          for _Mult in range (kMult):
            nb.unlink (lz)
            na.link (lz, LinkType)
            na.link (nb, LinkType)
          fin.append (na)
        else:
          unchecked = [z for z in cur]
          while len (unchecked) != 0:
            nb = random.choice (unchecked)
            if nb.getlinknode (na.id ()) is None:
              break
            else:
              inb = unchecked.index (nb)
              unchecked.pop (inb)
              nb = None
          if nb is None:
            nb = random.choice (cur)
          inb = cur.index (nb)
          cur.pop (inb)
          # connect the two nodes
          for _Mult in range (kMult):
            na.link (nb, LinkType)
          nxt.append (nb)
          nxt.append (na)
          # add to edges
          edges.append ((na, nb))
      cur = nxt
      nxt = fin
      fin = []

    for switch in this._switches:
      for link in switch.ports ():
        if link.n () == 1:
          this.add (link)


def test_generate (filename, target_design, verbose = False):
  # dut = DGXA100 ()
  # dut = SUA100 ()
  dut = SuperSpineA100 ()
  # dut = SpineGroupA100 ()
  # dut = SuperPodA100 ()
  output = dut.flatten ()
  print ('compute nodes (%s)' % (len (output['comp'])))
  if verbose:
    for node in output['comp']:
      print (node)
    print ()
  print ('switches nodes (%s)' % (len (output['switch'])))
  if verbose:
    for node in output['switch']:
      print (node)
    print ()
  cl = [link for link in output['link'] if link.n () == 2]
  print ('link nodes (%s)' % (len (cl)))
  if verbose:
    for node in output['link']:
      print (node)
  # read in configuratation
  yml = None
  try:
    with open (filename, 'r') as file:
      yml = yaml.load (file, Loader = Loader)
  except:
    yml = {}
  map = {}
  i = None
  for i, node in enumerate (output['comp']):
    map[node._id] = i
  # increment because enumerate does no overshoot
  i = i + 1
  for j, switch in enumerate (output['switch']):
    map[switch._id] = i + j
  if target_design not in yml.keys ():
    # add design
    yml[target_design] = {}
    design = yml[target_design]
    design['type'] = 'ns3:dev'
    design['ns3'] = {}
    design['ns3']['topology'] = {}
    # add dims
    design['dims'] = []
    design['dims'].append ({})
    design['dims'][0]['valid'] = True
    # add power
    design['power'] = {}
    design['power']['link'] = []
    design['power']['compute'] = []
    # add A100 compute by default
    design['power']['compute'].append ({
      'nodes': [],
      'idle': 50.0,
      'peak': 400.0
    })

  # set up compute
  design['dims'][0]['npus'] = len (output['comp'])
  # only considers 1st compute node
  design['dims'][0]['local-mem-bw'] = output['comp'][0].bw ()
  design['dims'][0]['peak-perf'] = output['comp'][0].perf ()

  # set up links
  design['ns3']['topology']['switches'] = len (output['switch'])
  design['ns3']['topology']['links'] = []
  linkpowers = {}
  for j, link in enumerate (output['link']):
    if link.bnode () is not None:
      l = {}
      l['node-a'] = map[link.anode ().id ()]
      l['node-b'] = map[link.bnode ().id ()]
      # put bandwidth in Gbps
      l['bandwidth'] = '%.1fGbps' % (link.bw () * 1e-9)
      # put in ms
      l['latency'] = '%.6fms' % (link.latency () * 1e3)
      # change later
      l['error-rate'] = 0.0
      design['ns3']['topology']['links'].append (l)
      # add power
      if type (link) not in linkpowers:
        lg = {
          'nodes': [],
          'idle': link.idle (),
          'peak': link.peak ()
        }
        linkpowers[type (link)] = len (design['power']['link'])
        design['power']['link'].append (lg)
      # append to node list
      design['power']['link'][linkpowers[type (link)]]['nodes'].append (j)

  # dump out contents
  with open (filename, 'w') as file:
    yaml.dump (yml, file, default_flow_style = False)

def main ():
  parser = argparse.ArgumentParser (
    prog = 'ASTRA-topology',
    description = '',
    epilog = '')
  parser.add_argument ('-c','--config', default='configuration.yaml')
  parser.add_argument ('-d','--design', required=True)
  parser.add_argument ('-v','--verbose', action='store_true', default = False)
  args = parser.parse_args (sys.argv[1:])
  test_generate (args.config, args.design, args.verbose)

if __name__ == '__main__':
  main ()
