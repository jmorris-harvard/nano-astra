#!/usr/bin/python3

import argparse
import numpy as np
import os
import pandas as pd
import re
import sys

def parse_packets (packets):
  data = {
    'timestamps': [],
    'sizes': [],
    'srcs': [],
    'dsts': []
  }
  for packet in packets:
    parsed_packet = parse_packet (packet)
    data['timestamps'].append (parsed_packet['timestamp'])
    data['sizes'].append (parsed_packet['size'])
    data['srcs'].append (parsed_packet['src'])
    data['dsts'].append (parsed_packet['dst'])
  return data

def parse_packet (packet):
  # pull out ipv4 header
  pattern = r'r' + r'\s*' + \
            r'(?P<timestamp>[0-9]+(\.[0-9]+)?(e-[0-9]+)?)' + r'\s*' + \
            r'ns3::PppHeader' + r'\s*' + r'\((?P<ppp>.*)\)' + r'\s*' + \
            r'ns3::Ipv4Header' + r'\s*' + r'\((?P<ip>.*)\)' + r'\s+' + 'ns3::((qbbHeader)|(UdpHeader))'
  m = re.match (pattern, packet)
  timestamp = float (m.group ('timestamp'))
  ipconfig = m.group ('ip')

  # parse ipv4 header
  pattern = r'tos\s*0x\w+\s*DSCP\s*Default\s*ECN\s*((Not-ECT)|(CE))\s*' + \
            r'ttl\s*[0-9]+\s*' + \
            r'id\s*[0-9]+\s*' + \
            r'protocol\s*[0-9]+\s*' + \
            r'offset\s*\(bytes\)\s*[0-9]+\s*' + \
            r'flags\s*\[[\w\s]+\]\s*' + \
            r'length:\s*(?P<size>[0-9]+)\s*' + \
            r'11\.0\.(?P<src>[0-9]+)\.1\s*>\s*' + \
            r'11\.0\.(?P<dst>[0-9]+)\.1'
  try:
    m = re.match (pattern, ipconfig)
    size = int (m.group ('size'))
    src = int (m.group ('src'))
    dst = int (m.group ('dst'))
  except:
    print (ipconfig)
    sys.exit ()
  
  return {
    'timestamp': timestamp,
    'size': size,
    'src': src,
    'dst': dst
  }


def main ():
  parser = argparse.ArgumentParser(
    prog='ASTRA-ns3-Packet-Parser',
    description='',
    epilog='')
  parser.add_argument('-l', '--location', required = True)
  parser.add_argument('-p', '--prefix', required = True)
  parser.add_argument('-o', '--output', default = 'packet-parser-out.csv')
  parser.add_argument('-a', '--append', action = 'store_true', default = False)
  args = parser.parse_args (sys.argv[1:])

  # write output header
  access = 'a' if args.append else 'w'
  with open (args.output, access) as csv:
    # timestamp,node,link,src,dst,size
    csv.write ('timestamp,node,link,src,dst,size\n')

  pattern = args.prefix + r'-(?P<node>[0-9]+)-(?P<link>[0-9]+).tr'
  traces = [{'name': t, 'regex': re.match (pattern, t)} for t in os.listdir (args.location) if re.match (pattern, t)]
  for trace in traces:
    filename = os.path.join (args.location, trace['name'])
    print ('Parsing (%s)...' % filename)
    with open (filename) as tr:
      packets = tr.read ().splitlines ()
      data = parse_packets (packets)
      # parse into csv
      nsamples = len (data['timestamps'])
      print ('\tContained (%d) packet events' % (nsamples))
      nodes = [trace['regex'].group('node') for _ in range (nsamples)]
      # links are 1 indexed lets change that
      links = [int (trace['regex'].group('link')) - 1 for _ in range (nsamples)]
      # timestamp,node,link,src,dst,size
      rows = zip (data['timestamps'],nodes,links,data['srcs'],data['dsts'],data['sizes'])
      with open (args.output, 'a') as csv:
        for row in rows:
          r = [str (item) for item in row]
          csv.write (','.join (r) + '\n')

if __name__ == '__main__':
  main ()
