#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys
import yaml

try:
  from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
  from yaml import Loader, Dumper

def get_per_node_power (filename, compute, timestep = 1e-9):
  # expects file with tracker syntax using Roofline model in astra-sim
  lines = None
  with open (filename, 'r') as tracker:
    pattern = r'\[tracker\] compute'
    lines = [line for line in tracker.read ().splitlines () if re.match (pattern, line)]

  report = {}
  pattern = r'\[tracker\] compute,' + \
            r'(?P<workload>[0-9]+),' + \
            r'(?P<node>[0-9]+),' + \
            r'(?P<start>[0-9]+),' + \
            r'(?P<ops>[0-9]+),' + \
            r'(?P<tensor>[0-9]+),' + \
            r'(?P<intensity>[0-9]+\.?([0-9]+)?(e[+-][0-9]+)?),' + \
            r'(?P<perf>[0-9]+\.?([0-9]+)?(e[+-][0-9]+)?),' + \
            r'(?P<limit>[0-9]+\.?([0-9]+)?(e[+-][0-9]+)?),' + \
            r'(?P<runtime>[0-9]+)'
  for line in lines:
    # print (line)
    m = re.match (pattern, line)
    node = int (m.group ('node'))
    spec = compute.get (node, compute['default'])
    if node not in report.keys ():
      report[node] = []
    start = float (m.group ('start'))
    runtime = float (m.group ('runtime'))
    perf = float (m.group ('perf'))
    peak = float (m.group ('limit'))
    # compute power and append
    power = spec['idle'] + spec['scale'] * (perf / peak) # simple dynamic frequency scaling
    report[node].append ((start * timestep, (start + runtime) * timestep, power))
  # sort by start time
  for node in report.keys ():
    report[node].sort (key = lambda x: x[0])
  # should not have overlapping communication
  # create df
  reportdict = {}
  reportdict['node'] = []
  reportdict['start'] = []
  reportdict['end'] = []
  reportdict['power'] = []
  for node in report.keys ():
    for start, end, power in report[node]:
      reportdict['node'].append (node)
      reportdict['start'].append (start)
      reportdict['end'].append (end)
      reportdict['power'].append (power)
  return pd.DataFrame.from_dict (reportdict)

def get_per_port_power (filename, link):
  lines = None
  with open (filename, 'r') as packets:
    # expects file with parsed packet syntax used in parse-ns3-packets
    lines = packets.read ().splitlines ()[1:] # skip header

  report = []
  pattern = r'(?P<node>[0-9]+),' + \
            r'(?P<port>[0-9]+),' + \
            r'(?P<uid>[0-9]+),' + \
            r'(?P<direction>(Rx)|(Tx)),' + \
            r'(?P<arrival>[0-9]+\.?([0-9]+)?(e[+-][0-9]+)?),' + \
            r'(?P<departure>[0-9]+\.?([0-9]+)?(e[+-][0-9]+)?),' + \
            r'(?P<size>[0-9]+)'

  for line in lines:
    m = re.match (pattern, line)
    node = int (m.group ('node'))
    port = int (m.group ('port'))
    spec = link[node][port]
    # initialize link consumption
    arrival = float (m.group ('arrival'))
    departure = float (m.group ('departure'))
    report.append ((node, port, arrival, departure, spec['peak']))
  # sort by start times
  report.sort (key = lambda x: x[0])
  reportdict = dict ()
  reportdict['node'] = []
  reportdict['port'] = []
  reportdict['start'] = []
  reportdict['end'] = []
  reportdict['power'] = []
  for node, port, start, end, power in report:
    reportdict['node'].append (node)
    reportdict['port'].append (port)
    reportdict['start'].append (start)
    reportdict['end'].append (end)
    reportdict['power'].append (power)
  return pd.DataFrame.from_dict (reportdict)

def get_per_link_power_legacy (filename, link):
  lines = None
  with open (filename, 'r') as packets:
    # expects file with parsed packet syntax used in parse-ns3-packets
    lines = packets.read ().splitlines ()[1:] # skip header

  report = {}
  pattern = r'(?P<recvtime>[0-9]+\.?([0-9]+)?(e[+-][0-9]+)?),' + \
            r'(?P<node>[0-9]+),' + \
            r'(?P<link>[0-9]+),' + \
            r'(?P<src>[0-9]+),' + \
            r'(?P<dst>[0-9]+),' + \
            r'(?P<size>[0-9]+)'
  for line in lines:
    m = re.match (pattern, line)
    node = int (m.group ('node'))
    l = int (m.group ('link'))
    a, b = link['mappings'][node][l]
    a, b = min (a, b), max (a, b)
    spec = link[a][b]
    index = spec['id']
    # initialize link consumption
    if index not in report.keys ():
      report[index] = []
    recv = float (m.group ('recvtime'))
    latency = float (spec['latency']) + float (float (m.group ('size')) / spec['bandwidth'])
    peak = float (spec['peak'])
    report[index].append ((recv - latency, recv, peak, a, b))
  # sort by start times
  for index in report.keys ():
    report[index].sort (key = lambda x: x[0])
  # resolve overlapping communication
  for index in report.keys ():
    updated = True
    while updated:
      updated = False
      new = []
      # merge adjacent overlapping operations
      i = 0
      while i < len (report[index]):
        if i < len(report[index]) - 1 and report[index][i][1] >= report[index][i + 1][0]:
          # merge cells
          peak = max (report[index][i][2], report[index][i + 1][2])
          new.append ((
            report[index][i][0], 
            report[index][i + 1][1], 
            peak, 
            report[index][i][3],
            report[index][i][4]))
          updated = True
          i = i + 2
        else:
          new.append (report[index][i])
          i = i + 1
      # overwrite old value
      report[index] = new
  reportdict = {}
  reportdict['link'] = []
  reportdict['start'] = []
  reportdict['end'] = []
  reportdict['power'] = []
  reportdict['node-a'] = []
  reportdict['node-b'] = []
  for index in report.keys ():
    for start, end, power, a, b in report[index]:
      reportdict['link'].append (index)
      reportdict['start'].append (start)
      reportdict['end'].append (end)
      reportdict['power'].append (power)
      reportdict['node-a'].append (a)
      reportdict['node-b'].append (b)
  return pd.DataFrame.from_dict (reportdict)

def parse_prefix (prefix_string):
  conv = 1.0
  if 'n' in prefix_string:
    conv = 1e-9
  elif 'u' in prefix_string:
    conv = 1e-6
  elif 'm' in prefix_string:
    conv = 1e-3
  elif 'k' in prefix_string:
    conv = 1e+3
  elif 'M' in prefix_string:
    conv = 1e+6
  elif 'G' in prefix_string:
    conv = 1e+9
  else:
    print ('unsupported scale given (%s)' % (conv))
    sys.exit (1)
  return conv

def parse_number_string (latency_string):
  # pattern = r'(?P<number>[0-9]+(.[0-9]+)?)'
  pattern = r'(?P<number>[0-9]+(.[0-9]+)?)(?P<conv>\w+)?'
  m = re.match (pattern, latency_string)
  number = float (m.group ('number'))
  conv = None
  try:
    conv = parse_prefix (m.group ('conv'))
  except:
    print ('failed to parse number (%s)' % (latency_string))
    sys.exit ()
  return  number * conv

def parse_config (config_filename, target_design, topology_filename):
  # read design
  yml = None
  with open (config_filename, 'r') as file:
    yml = yaml.load (file, Loader = Loader)
  design = yml[target_design]
  
  # get power specs
  power = {}
  
  # get compute power description
  power['compute'] = {}
  # default consumption is none
  power['compute']['default'] = {}
  power['compute']['default']['idle'] = 0
  power['compute']['default']['peak'] = 0
  power['compute']['default']['scale'] = 0
  for group in design['power']['compute']:
    if len (group['nodes']) == 0:
      # 0 length group is chosen as default
      power['compute']['default']['idle'] = float (group['idle'])
      power['compute']['default']['peak'] = float (group['peak'])
      power['compute']['default']['scale'] = power['compute']['default']['peak'] - power['compute']['default']['idle']
    for node in group['nodes']:
      # get each description
      node = int (node)
      power['compute'][node] = {}
      power['compute'][node]['idle'] = float(group['idle'])
      power['compute'][node]['peak'] = float(group['peak'])
      power['compute'][node]['scale'] = power['compute'][node]['peak'] - power['compute'][node]['idle']

  # get link power description
  power['link'] = {}
  # default consumption is none
  power['link']['default'] = {}
  power['link']['default']['idle'] = 0
  power['link']['default']['peak'] = 0
  power['link']['default']['scale'] = 0
  for group in design['power']['link']:
    if len (group['links']) == 0:
      # 0 length group is chosen as default
      power['link']['default']['idle'] = float (group['idle'])
      power['link']['default']['peak'] = float (group['peak'])
      power['link']['default']['scale'] = power['link']['default']['peak'] - power['link']['default']['idle']
    for link in group['links']:
      # get each description
      # order lowest id 1st
      a = int(link['node-a'])
      b = int(link['node-b'])
      if a not in power['link'].keys ():
        power['link'][a] = []
      if b not in power['link'].keys ():
        power['link'][b] = []
      record = dict ()
      record['idle'] = float(group['idle'])
      record['peak'] = float(group['peak'])
      record['scale'] = record['peak'] - record['idle']
      power['link'][a].append (record)
      power['link'][b].append (record)

  # get other power draw
  power['other'] = list ()
  if design['power'].get ('other', None) is not None:
    other = design['power']['other']
    for consumer in other:
      op = dict ()
      op['name'] = consumer['name']
      op['idle'] = consumer['idle']
      power['other'].append (op)
  return power

def parse_config_legacy (config_filename, target_design, topology_filename):
  # read design
  yml = None
  with open (config_filename, 'r') as file:
    yml = yaml.load (file, Loader = Loader)
  design = yml[target_design]
  
  # get power specs
  power = {}
  
  # get compute power description
  power['compute'] = {}
  # default consumption is none
  power['compute']['default'] = {}
  power['compute']['default']['idle'] = 0
  power['compute']['default']['peak'] = 0
  power['compute']['default']['scale'] = 0
  for group in design['power']['compute']:
    if len (group['nodes']) == 0:
      # 0 length group is chosen as default
      power['compute']['default']['idle'] = float (group['idle'])
      power['compute']['default']['peak'] = float (group['peak'])
      power['compute']['default']['scale'] = power['compute']['default']['peak'] - power['compute']['default']['idle']
    for node in group['nodes']:
      # get each description
      node = int (node)
      power['compute'][node] = {}
      power['compute'][node]['idle'] = float(group['idle'])
      power['compute'][node]['peak'] = float(group['peak'])
      power['compute'][node]['scale'] = power['compute'][node]['peak'] - power['compute'][node]['idle']

  # get link power description
  power['link'] = {}
  # default consumption is none
  power['link']['default'] = {}
  power['link']['default']['idle'] = 0
  power['link']['default']['peak'] = 0
  power['link']['default']['scale'] = 0
  for group in design['power']['link']:
    if len (group['links']) == 0:
      # 0 length group is chosen as default
      power['link']['default']['idle'] = float (group['idle'])
      power['link']['default']['peak'] = float (group['peak'])
      power['link']['default']['scale'] = power['link']['default']['peak'] - power['link']['default']['idle']
    for link in group['links']:
      # get each description
      # order lowest id 1st
      a = int(link['node-a'])
      b = int(link['node-b'])
      a, b = min (a, b), max (a, b)
      if a not in power['link'].keys ():
        power['link'][a] = {}
      power['link'][a][b] = {}
      power['link'][a][b]['idle'] = float(group['idle'])
      power['link'][a][b]['peak'] = float(group['peak'])
      power['link'][a][b]['scale'] = power['link'][a][b]['peak'] - power['link'][a][b]['idle']

  # also need topology description
  power['link']['mappings'] = {}
  with open (topology_filename, 'r') as topology:
    # grab link information
    lines = topology.read().splitlines ()[2:]
    pattern = r'(?P<nodea>[0-9]+)\s*' + \
              r'(?P<nodeb>[0-9]+)\s*' + \
              r'(?P<bandwidth>[0-9]+\.?([0-9]+)?\w*)\s*' + \
              r'(?P<latency>[0-9]+\.?([0-9]+)?\w*)\s*' + \
              r'(?P<errorrate>[0-9]+\.?([0-9]+)?\w*)\s*'
    for i,link in enumerate (lines):
      m = re.match (pattern, link)
      a = int (m.group ('nodea'))
      b = int (m.group ('nodeb'))
      a, b = min (a, b), max (a, b)
      # resolve missing links
      if a not in power['link'].keys ():
        power['link'][a] = {}
      if b not in power['link'][a].keys ():
        power['link'][a][b] = {}
        power['link'][a][b]['idle'] = power['link']['default']['idle']
        power['link'][a][b]['peak'] = power['link']['default']['peak']
        power['link'][a][b]['scale'] = power['link']['default']['scale']
      power['link'][a][b]['latency'] = parse_number_string (m.group ('latency'))
      power['link'][a][b]['bandwidth'] = parse_number_string (m.group ('bandwidth'))
      # map (node, link) to (node, node)
      # assumes links are added in order
      mapping = (a, b)
      if a not in power['link']['mappings'].keys ():
        power['link']['mappings'][a] = []
      power['link']['mappings'][a].append (mapping)
      if b not in power['link']['mappings'].keys ():
        power['link']['mappings'][b] = []
      power['link']['mappings'][b].append (mapping)
      power['link'][a][b]['id'] = i
  return power

def create_range (start, end, timestep, value):
  steps = int ((end - start) / timestep)
  x = np.arange (0, steps) * timestep
  x = x + start
  y = np.ones (steps) * value
  return x, y

def plot_power_compute (compute, compute_df, end, start = 0.0, timestep = 1e-3):
  unique = compute_df['node'].unique ()
  plotting = list ()
  plt.ticklabel_format(style='sci', axis='x', scilimits=(-2,2))
  for node in unique:
    data = compute_df[compute_df['node'] == node]
    spec = compute.get (node, compute['default'])
    samples = int ((end - start) / timestep)
    X = np.linspace (start, end, samples)
    Y = np.ones (samples) * spec['idle']
    # iterate through operations
    for i in range (len (data)):
      arrival = data.iloc[i]['start']
      departure =  data.iloc[i]['end']
      if arrival > end or departure < start:
        continue
      current = 0
      for j, val in enumerate (X[current:]):
        if val < arrival:
          continue
        elif val > departure:
          current = current + j
          break
        else:
          Y[current + j] = data.iloc[i]['power']
    plot = {}
    plot['X'] = X
    plot['Y'] = Y
    plot['node'] = node
    plotting.append (plot)
    fig, ax = plt.subplots ()
    ax.plot (X, Y)
    ax.set_ylabel ('Power (W)')
    ax.set_xlabel ('Time (s)')
    ax.set_title ('Node %d' % (node))
    ax.set_xlim (start, end)
    fig.savefig ('compute-%d.png' % (node))
    plt.close ()
  # plot total compute
  if len (plotting) > 0:
    samples = int ((end - start) / timestep)
    X = np.linspace (start, end, samples)
    Y = np.zeros (samples)
    for plot in plotting:
      Y = Y + plot['Y']
    fig, ax = plt.subplots ()
    ax.plot (X, Y)
    ax.set_ylabel ('Power (W)')
    ax.set_xlabel ('Time (s)')
    ax.set_title ('Total Compute Power')
    ax.set_xlim (start, end)
    fig.savefig ('compute-total.png')
    plt.close ()
  return plotting, X, Y 

def plot_power_link (link, link_df, end, start = 0.0, timestep = 1e-6):
  unique_node = link_df['node'].unique ()
  plotting = list ()
  plt.ticklabel_format(style='sci', axis='x', scilimits=(-2,2))
  for node in unique_node:
    data_node = link_df[link_df['node'] == node]
    unique_port = data_node['port'].unique ()
    for port in unique_port:
      data_port = data_node[data_node['port'] == port]
      spec = None
      if node not in link or port not in link[node]:
        spec = link['default']
      else:
        spec = link[node][port]
      samples = int ((end - start) / timestep)
      X = np.linspace (start, end, samples)
      Y = np.ones (samples) * spec['idle']
      # iterate through operations
      current = 0
      for i in range (len (data_port)):
        arrival = data_port.iloc[i]['start']
        departure =  data_port.iloc[i]['end']
        if arrival > end or departure < start:
          continue
        for j, val in enumerate (X[current:]):
          if val < arrival:
            continue
          elif val > departure:
            current = current + j
            break
          else:
            Y[current + j] = data_port.iloc[i]['power']
      plot = dict ()
      plot['X'] = X
      plot['Y'] = Y
      plot['node'] = node
      plot['port'] = port
      plotting.append (plot)
      fig, ax = plt.subplots ()
      ax.plot (X, Y)
      ax.set_ylabel ('Power (W)')
      ax.set_xlabel ('Time (s)')
      ax.set_title ('Node (%d) - Port (%d)' % (node, port))
      ax.set_xlim (start, end)
      fig.savefig ('link-%d-%d.png' % (node, port))
      plt.close ()
  
  # plot total link communication
  if len (plotting) > 0:
    samples = int ((end - start) / timestep)
    X = np.linspace (start, end, samples)
    Y = np.zeros (samples)
    for plot in plotting:
      Y = Y + plot['Y']
    fig, ax = plt.subplots ()
    ax.plot (X, Y)
    ax.set_ylabel ('Power (W)')
    ax.set_xlabel ('Time (s)')
    ax.set_title ('Total Communication Power')
    ax.set_xlim (start, end)
    fig.savefig ('communication-total.png')
    plt.close ()
  return plotting, X, Y

def plot_power_link_legacy (link, link_df, end, start = 0.0, timestep = 1e-6):
  unique = link_df['link'].unique ()
  fig, axes = plt.subplots (nrows = len(unique) + 1, figsize = (6, (len (unique) + 1) * 3))
  plotting = {}
  for i, l in enumerate (unique):
    data = link_df[link_df['link'] == l]
    a, b = data['node-a'].iloc[0], data['node-b'].iloc[0]
    spec = link[a][b]
    X = []
    Y = []
    last = start
    final = start
    # iterate through operations and plot
    for j in range (len (data)):
      # check start is in range
      if data.iloc[j]['start'] > end:
        # leave idle to post loop code
        break
      # check end is in range
      if data.iloc[j]['end'] < start:
        # ignore this node
        continue
      # move to node start
      if data.iloc[j]['start'] > last:
        x, y = create_range (last, data.iloc[j]['start'], timestep, 0) # idle --> 0
        X.extend (x)
        Y.extend (y)
      # check end still in range
      final = data.iloc[j]['end']
      if final > end:
        # extend power to time end
        final = end
      x, y = create_range (data.iloc[j]['start'], final, timestep, data.iloc[j]['power'])
      X.extend (x)
      Y.extend (y)
      # update last
      last = final
    # add tail
    x, y = None, None
    if len (X) == 0:
      # no computation on this node so populate whole range
      x, y = create_range (start, end, timestep, 0) # idle --> 0
    elif final != end:
      # computation did not extend beyond end so add some padding
      x, y = create_range (final, end, timestep, 0) # idle --> 0
    else:
      # no need to pad
      x, y = [], []
    X.extend (x)
    if X[0] != 0.0:
      print ('error occurred during plotting, try a smaller timescale')
      sys.exit ()
    Y.extend (y)
    plotting[l] = {}
    plotting[l]['X'] = X
    plotting[l]['Y'] = Y
    plotting[l]['link'] = f'{a}_{b}'
    # update Y to true idle value
    Y = [spec['idle'] if y == 0 else y for y in Y]
    axes[i].plot (X, Y)
    axes[i].set_ylabel ('Link (%d) Power (W)' % (l))
    axes[i].set_xlim (start, end)
  # plot total link compute
  if len (unique) > 0:
    Y = np.zeros (shape = len (Y))
    for l in plotting.keys ():
      trunc = min (len (Y), len (plotting[l]['Y']))
      Y = Y[:trunc] + np.array (plotting[l]['Y'])[:trunc]
    X = X[:len(Y)]
    axes[i + 1].plot (X, Y)
    axes[i + 1].set_ylabel ('Total Link Power (W)')
    axes[i + 1].set_xlim (start, end)
  fig.suptitle ('link power (W)')
  plt.ticklabel_format(style='sci', axis='x', scilimits=(-2,2))
  plt.savefig ('link.png')
  return plotting, X, Y

def main (): 
  parser = argparse.ArgumentParser(
    prog='ASTRA-power',
    description='',
    epilog='')
  parser.add_argument('-l', '--link', required = True)
  parser.add_argument('-p', '--compute', required = True)
  parser.add_argument('-t', '--topology', required = True)
  parser.add_argument('-c', '--configuration', required = True)
  parser.add_argument('-d', '--design', required = True)
  parser.add_argument('-e', '--end', type = float, default = None)
  parser.add_argument('--timestep', type = float, default = 10e-9)
  parser.add_argument('--repetitions', type = float, default = 1)
  parser.add_argument('--legacy', action = 'store_true', default = False)
  args = parser.parse_args (sys.argv[1:])

  # parse all data
  power = None
  if args.legacy:
    power = parse_config_legacy (args.configuration, args.design, args.topology)
  else:
    power = parse_config (args.configuration, args.design, args.topology)
  # count nodes and switches
  compute_df = get_per_node_power (args.compute, power['compute'])
  compute_df.to_csv ('compute.csv')
  link_df = None
  if args.legacy:
    link_df = get_per_link_power_legacy (args.link, power['link'])
  else:
    link_df = get_per_port_power (args.link, power['link'])
  link_df.to_csv ('link.csv')

  # determine end
  end = None
  if not args.end:
    # use output df to determine end
    end_compute = max (compute_df['end']) if len (compute_df['end']) > 0 else 0.0 
    end_link = max (link_df['end']) if len (link_df['end']) > 0 else 0.0
    end = max (end_compute, end_link)
  else:
    end = float (args.end)

  # spit out some plots
  c, cx, cy = plot_power_compute (power['compute'], compute_df, end = end, timestep = args.timestep)
  l, lx, ly = plot_power_link (power['link'], link_df, end = end, timestep = args.timestep)

  # -- sum everything to get total
  # ---- if no data just quit
  if len (cx) == 0 and len (lx) == 0:
    print ('No data to report')
    sys.exit ()

  # add other line
  oe_p = 0.0
  for other in power['other']:
    oe_p = oe_p + other['idle']
  oy = np.ones (shape = cy.shape) * oe_p

  X = cx
  Y = cy + ly + oy
  fig, ax = plt.subplots ()
  ax.plot (X, Y)
  ax.set_xlabel ('Time (s)')
  ax.set_ylabel ('Aggregate Power (W)')
  ax.set_title ('Total Aggregate Power')
  fig.savefig ('total.png')
  plt.close ()

  '''
  # animation
  # create list on each time step and dump
  animation_list = []
  # create utilization at the same time
  utilization = {}
  for i in range (len (X)):
    animation_point = ['']
    for lkey in l.keys ():
      utilization[l[lkey]['link']] = utilization.get (l[lkey]['link'], 0)
      if int (l[lkey]['Y'][i]) != 0:
        animation_point.append (l[lkey]['link'])
        utilization[l[lkey]['link']] = utilization.get (l[lkey]['link'], 0) + 1
    for ckey in c.keys ():
      utilization[ckey] = utilization.get (ckey, 0)
      if int (c[ckey]['Y'][i]) != 0:
        animation_point.append (str (ckey))
        utilization[ckey] = utilization.get (ckey, 0) + 1
    animation_list.append (','.join (animation_point) + '\n')
  with open ('animation.csv', 'w') as csv:
    for item in animation_list:
      csv.write (item)
  # dump utilization numbers
  with open ('utilization.csv','w') as util:
    util.write ('node/link,utilization\n')
    for ukey in utilization.keys ():
      util.write (','.join ([str (ukey), str(utilization[ukey] / xylen)]) + '\n')
  '''

  # report compute energy
  ce_j = sum (cy) * args.timestep * args.repetitions
  ce_mwh = (ce_j / 3600.0) * 1.0e-3
  print ('Total Compute Energy (%f J), (%f kWh)' % (ce_j, ce_mwh))
  # report link energy
  le_j = sum (ly) * args.timestep * args.repetitions
  le_mwh = (le_j / 3600.0) * 1.0e-3
  print ('Total Link Energy (%f J), (%f kWh)' % (le_j, le_mwh))
  # report other idle energy
  oe_j = oe_p * end * args.repetitions
  oe_mwh = (oe_j / 3600.0) * 1.0e-3
  print ('Total Other Component Energy (%f J), (%f kWh)' % (oe_j, oe_mwh))
  # report total energy
  te_j = ce_j + le_j + oe_j
  te_mwh = ce_mwh + le_mwh + oe_mwh
  print ('Total Combined Energy (%f J), (%f kWh)' % (te_j, te_mwh))
  with open ('total.csv', 'w') as csv:
    csv.write ('component,energy(J),energy(kWh)\n')
    csv.write ('compute,%f,%f\n' % (ce_j, ce_mwh))
    csv.write ('communication,%f,%f\n' % (le_j, le_mwh))
    csv.write ('all,%f,%f\n' % (te_j, te_mwh))

if __name__ == '__main__':
  main ()
