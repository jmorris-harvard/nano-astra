#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys

def parse_log (logfile, verbose = False):
  lines = None
  with open (logfile) as log:
    lines = log.read ().splitlines ()
  
  data = {
    'action': [],
    'npu': [],
    'tick': [],
    'node': [],
    'task': [],
    'type': []
  }
  type_mapping = {
    1: 'unknown',
    2: 'unknown',
    3: 'unknown',
    4: 'compute',
    5: 'communication', # send
    6: 'communication', # recv
    7: 'communication' # collective
  }
  name_mapping = {
    1: 'unknown',
    2: 'unknown',
    3: 'unknown',
    4: 'COMP_NODE',
    5: 'COMM_SEND_NODE', # send
    6: 'COMM_RECV_NODE', # recv
    7: 'COMM_COLL_NODE' # collective
  }

  # parse lines one at a time
  output = []
  for line in lines:
    re_begin = r'\[(?P<date>\d+-\d+-\d+)' + r'\s*' + \
        r'(?P<time>\d+:\d+:\d+\.\d+)\]' + r'\s*' + \
        r'\[(?P<layer>[\w:]+)\]' + r'\s*' + \
        r'\[(?P<severity>\w+)\]' + r'\s*'
    match = re.match (re_begin, line, re.ASCII)
    if verbose:
      print ('layer=' + match.group ('layer'), end=' ')
      print ('severity=' + match.group ('severity'), end=' ')
    if not match.group ('layer') == 'workload' or \
       not match.group ('severity') == 'debug':
      continue
    line = re.sub (re_begin, '', line, re.ASCII)
    re_end = r'(?P<action>\w+),?' + r'\s*' + \
        r'sys->id=(?P<npu>\w+),?' + r'\s*' + \
        r'tick=(?P<tick>\w+),?' + r'\s*' + \
        r'node->id=(?P<node>\w+),?' + r'\s*' + \
        r'node->name=(?P<task>\w+),?' + r'\s*' + \
        r'node->type=(?P<type>\w+),?' + r'\s*'
    match = re.match (re_end, line, re.ASCII)
    # rebuild line using match
    newline = match.group ('action') + ','  + \
              'npuId=' + match.group ('npu') + ','  + \
              'cycle=' + match.group ('tick') + ','  + \
              'nodeId=' + match.group ('node') + ','  + \
              'nodeName=' + name_mapping[int (match.group ('type'))] + '_' + match.group ('task')
    output.append (newline)
    if verbose:
      print ('action=' + match.group ('action'), end=' ')
      print ('npu=' + match.group ('npu'), end=' ')
      print ('tick=' + match.group ('tick'), end=' ')
      print ('node=' + match.group ('node'), end=' ')
      print ('task=' + match.group ('task'), end=' ')
      print ('type=' + match.group ('type'))
    for key in data.keys ():
      if key == 'type':
        data[key].append (type_mapping[int(match.group (key))])
      elif key == 'npu' or \
           key == 'tick' or \
           key == 'node':
        data[key].append (int (match.group(key)))
      else:
        data[key].append (match.group (key))

  return pd.DataFrame.from_dict (data), output

def build_timeline (df, out_filename = 'timeline.png', sep = False):
  height = 0.8
  spacing = 0.8
  npus = df['npu'].unique ()
  num_npus = len(npus)
  tasks = df['type'].unique ()
  num_tasks = len(tasks)
  task_mapping = {}
  for i, task in enumerate (tasks):
    task_mapping[task] = i + 1
  gantt = {}
  node_mapping = {}
  for npu in npus:
    gantt[npu] = {}
    gantt[npu]['task'] = []
    gantt[npu]['id'] = []
    gantt[npu]['issue'] = []
    gantt[npu]['callback'] = []
    node_mapping[npu] = {}
  for i in range (len (df)):
    row = df.iloc[i]
    npu = row['npu']
    if row['action'] == 'issue':
      gantt[npu]['task'].append (row['type'])
      gantt[npu]['id'].append (row['node'])
      gantt[npu]['issue'].append (row['tick'])
      gantt[npu]['callback'].append (-1)
      node_mapping[npu][row['node']] = len(gantt[npu]['task']) - 1
    elif row['action'] == 'callback':
      gantt[npu]['callback'][node_mapping[npu][row['node']]] = row['tick']
    else:
      print ('error: unknown action taken (%s), exiting...')
      sys.exit (1)
  axs = None
  figs = None
  if sep:
    tuples = [plt.subplots(figsize = (10, 3)) for i in range (num_npus)]
    figs = [x[0] for x in tuples]
    axs = [x[1] for x in tuples]
  else:
    fig, axs = plt.subplots (nrows = num_npus, ncols = 1, sharex = True, figsize = (10, 2 * num_npus))
    figs = [ fig ]
  for i, npu in enumerate (npus):
    timeline_data = pd.DataFrame.from_dict (gantt[npu])
    y = None
    if isinstance (timeline_data['task'][0], int):
      y = np.array (timeline_data['task']) * spacing
    else:
      y = timeline_data['task']
    width = np.array (timeline_data['callback']) - np.array (timeline_data['issue'])
    left = np.array (timeline_data['issue'])
    axs[i].barh (y, width, height, left, edgecolor = 'black') 
    if sep:
      axs[i].set_title ('%s timeline' % npu)
      axs[i].set_xlabel ('time (cycles)')
    else:
      axs[i].set_ylabel ('%s' % npu)
      figs[0].suptitle ('timeline all npus')
      figs[0].supxlabel ('time (cycles)')
  prepend = ''
  for i, fig in enumerate (figs):
    if sep:
      fig.savefig (str(i) + '_' + out_filename)
    else:
      fig.savefig (out_filename)


def main ():
  parser = argparse.ArgumentParser(
    prog='ASTRA-timeline',
    description='',
    epilog='')
  parser.add_argument('-i', '--input', default = 'log.log')
  parser.add_argument('-s', '--separate', action = 'store_true', default = False)
  parser.add_argument('--build', action = 'store_true', default = False)
  parser.add_argument('-o', '--output', default = 'parsed.log')
  args = parser.parse_args (sys.argv[1:])
  df, chakra = parse_log (args.input)
  df.to_csv (args.input.replace ('.log', '.csv'))
  with open (args.output, 'w') as out:
    out.writelines ([line + '\n' for line in chakra])
  if args.build:
    build_timeline (df, out_filename = args.output.replace ('.log', '.png'), sep = args.separate)

if __name__ == '__main__':
  main ()
