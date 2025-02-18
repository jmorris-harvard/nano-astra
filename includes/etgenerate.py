#!/usr/bin/python3

import io
import os

from astragen import (
  ChakraNode,
  ChakraAttr,
  encode,
  COMP_NODE,
  COMM_COLL_NODE,
  ALL_REDUCE,
  COMM_SEND_NODE,
  COMM_RECV_NODE
)

def generate (index, nodes, et):
  for node in nodes:
    # create ns
    n = ChakraNode ()

    # required fields
    n.id = node['id']
    n.name = node['name']
    n.type = node['type']
    n.data_deps.extend (node['data-deps'])

    # optionals and attributes
    # -- optionals 
    if 'duration' in node.keys ():
      n.duration_micros = node['duration']
    # -- attributes
    if 'num-ops' in node.keys ():
      n.attr.append (ChakraAttr (name = 'num_ops', int64_val = node['num-ops']))
    if 'collective' in node.keys ():
      n.attr.append (ChakraAttr (name = 'comm_type', int64_val = node['collective']))
    if 'priority' in node.keys ():
      n.attr.append (ChakraAttr (name = 'comm_priority', int32_val = node['priority']))
    if 'src' in node.keys ():
      n.attr.append (ChakraAttr (name = 'comm_src', int32_val = node['src']))
    if 'dst' in node.keys ():
      n.attr.append (ChakraAttr (name = 'comm_dst', int32_val = node['dst']))
    if 'tag' in node.keys ():
      n.attr.append (ChakraAttr (name = 'comm_tag', int32_val = node['tag']))

    # special cases
    # -- computation
    if n.type == COMP_NODE:
      if 'size' in node.keys ():
        n.attr.append (ChakraAttr (name = 'tensor_size', uint64_val = node['size']))
    # -- communication
    if n.type in (COMM_COLL_NODE, COMM_SEND_NODE, COMM_RECV_NODE):
      n.attr.append (ChakraAttr (name = 'is_cpu_op', bool_val = False))
      if 'size' in node.keys ():
        n.attr.append (ChakraAttr (name = 'comm_size', int64_val = node['size']))

    # add n to et
    encode (et, n)

if __name__ == "__main__":
  pass
