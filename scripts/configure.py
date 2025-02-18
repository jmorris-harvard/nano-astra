#!/usr/bin/python3

import argparse
import glob
import importlib
import os
import re
import shutil
import subprocess
import sys
import yaml

import etgenerate
from astragen import encode, GlobalMetadata
try:
  from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
  from yaml import Loader, Dumper

class Mappings ():
  Network = {
    'ring': 'Ring',
    'switch': 'Switch',
    'fully-connected': 'FullyConnected'
  }
  System = {
    'ring': 'ring',
    'halving-doubling': 'halvingDoubling',
    'direct': 'direct'
  }
  Ns3Network = {
    'npus': 'logical-dims'
  }

def generate_config_ns3 (filename, design, target_design, target_workload, binary, overwrite = True):
  # network
  network = {
    # -- required arguments
    'req': {
      'npus': [] 
    },
    # -- optional arguments
    'opt': {
    }
  }

  # system
  system = {
    # -- required arguments
    'req': {
      'all-reduce': [],
      'all-gather': [],
      'reduce-scatter': [],
      'all-to-all': []
    },
    # -- optional arguments
    'opt': {
      'scheduling-policy': 'LIFO',
      'roofline-enabled': 1,
      'replay-only': 0,
      'endpoint-delay': 0,
      'preferred-dataset-splits': 4,
      'active-chunks-per-dimension': 1,
      'collective-optimization': 'localBWAware',
      'local-mem-bw': 50,
      'boost-mode': 0,
      'peak-perf': 0.001, # Measured in TFLOPS
    }
  }

  # ns3
  ns3 = {
    # -- required arguments
    'req': {
      'topology': [],
      'topology-file': None
    },
    # -- optional arguments
    'opt': { 
      'enable-qcn': 1,
      'use-dynamic-pfc-threshold': 1,
      'packet-payload-size': 1024,
      'flow-file': None,
      'trace-file': None,
      'trace-output-file': None,
      'fct-output-file': None,
      'pfc-output-file': None,
      'qlen-mon-file': None,
      'qlen-mon-start': 0,
      'qlen-mon-end': 20000,
      'simulator-stop-time': 4000000.0,
      'cc-mode': 12,
      'alpha-resume-interval': 1,
      'rate-decrease-interval': 1,
      'clamp-target-rate': 0,
      'rp-timer': 900,
      'ewma-gain': 0.00390625,
      'fast-recovery-times': 1,
      'rate-ai': '50Mb/s',
      'rate-hai': '100Mb/s',
      'min-rate': '100Mb/s',
      'dctcp-rate-ai': '1000Mb/s',
      'error-rate-per-link': 0.00,
      'l2-chunk-size': 4000,
      'l2-ack-interval': 1,
      'l2-back-to-zero': 0,
      'has-win': 1,
      'global-t': 0,
      'var-win': 1,
      'fast-react': 1,
      'u-target': 0.95,
      'mi-thresh': 0,
      'int-multi': 1,
      'multi-rate': 0,
      'sample-feedback': 0,
      'pint-log-base': 1.05,
      'pint-prob': 1.0,
      'nic-total-pause-time': 0,
      'rate-bound': 1,
      'ack-high_prio': 0,
      'link-down': [0, 0, 0],
      'enable-trace': 1,
      'kmax-map': [6, 25000000000, 400, 40000000000, 800, 100000000000, 1600, 200000000000, 2400, 800000000000, 3200, 1600000000000, 3200],
      'kmin-map': [6, 25000000000, 100, 40000000000, 200, 100000000000, 400, 200000000000, 600, 800000000000, 800, 1600000000000, 800],
      'pmax-map': [6, 25000000000, 0.2, 40000000000, 0.2, 100000000000, 0.2, 200000000000, 0.2, 800000000000, 0.2, 1600000000000, 0.2],
      'buffer-size': 32
    }
  }

  for dim in design['dims']:
    if dim['valid'] == True:
      # network
      # -- required arguments
      for arg in network['req'].keys ():
        network['req'][arg].append (dim[arg])
      # -- optional arguments
      for arg in network['opt'].keys ():
        if arg in dim.keys ():
          network['opt'][arg] = dim[arg]
      # system
      # -- required arguments
      for arg in system['req'].keys ():
        system['req'][arg].append (Mappings.System[dim[arg]])
      # -- optional argumens
      for arg in system['opt'].keys ():
        if arg in dim.keys ():
          system['opt'][arg] = dim[arg]
  # get npu count
  ncount = 1
  for nc in network['req']['npus']:
    ncount = ncount * nc

  # parse ns3 args
  # topologies
  if 'topology-file' in design['ns3'].keys ():
    ns3['req']['topology-file'] = design['ns3']['topology-file']
  else:
    # does not check validity
    ns3['req']['topology'] = design['ns3']['topology']

  # optional args
  for arg in ns3['opt'].keys ():
    if arg in design['ns3']:
      ns3['opt'][arg] = design['ns3'][arg]
      # nodes switches links
      # switch names

  # create project dir
  cwd = os.getcwd ()
  proj = os.path.join (cwd, target_design + '-' + target_workload)
  if os.path.isdir (proj) and overwrite:
    shutil.rmtree (proj)
  os.makedirs (proj, exist_ok = True)

  # make inputs
  inputs = os.path.join (proj, 'inputs')
  os.makedirs (inputs, exist_ok = True)

  outputs = os.path.join (proj, 'outputs')
  os.makedirs (outputs, exist_ok = True)

  # make network
  ncfg = os.path.join (inputs, 'network_cfg.json')
  with open (ncfg, 'w') as cfg:
    cfg.write ('{\n')
    # -- required
    for arg in network['req'].keys ():
      line = ', '.join (['"%s"' % str (i) for i in network['req'][arg]])
      cfg.write ('  "' + Mappings.Ns3Network[arg] + '": [ ' + line + ' ]\n')
    # -- optional
    for arg in network['opt'].keys ():
      cfg.write ('  "' + arg + '": "' + str (network['opt'].keys ()) + '",\n' )
    cfg.write ('}')

  # make ns3-config
  nscfg = os.path.join (inputs, 'ns3_cfg.txt')
  configuration_location = os.path.dirname (os.path.abspath (filename)) 
  with open (nscfg, 'w') as cfg:
    if ns3['req']['topology-file']:
      # copy topology file used elsewhere
      shutil.copy (os.path.join (configuration_location, ns3['req']['topology-file']), inputs)
      ns3['req']['topology-file'] = os.path.join (inputs, os.path.basename (ns3['req']['topology-file']))
    else:
      # create topology file
      nswitches = ns3['req']['topology']['switches']
      nnodes = ncount + nswitches
      nlinks = len (ns3['req']['topology']['links'])
      # if no links add 1 dummy link - stupid requirement (fix later)
      if nlinks == 0:
        ns3['req']['topology']['links'].append ({
          'node-a': 0,
          'node-b': 0,
          'bandwidth': '100Gbps',
          'latency': '0.0',
          'error-rate': 0
        })
        nlinks = 1
      ns3['req']['topology-file'] = os.path.join (inputs, 'ns3_topology.txt')
      with open (ns3['req']['topology-file'], 'w') as topfile:
        topfile.write ('%d %d %d\n' % (nnodes, nswitches, nlinks))
        topfile.write (' '.join ([str(i) for i in range (ncount, nnodes)]) + '\n')
        for link in ns3['req']['topology']['links']:
          topfile.write (' '.join ([
              str(link['node-a']),
              str(link['node-b']),
              str(link['bandwidth']),
              str(link['latency']),
              str(link['error-rate'])
          ]) + '\n')

    if ns3['opt']['trace-file']:
      # copy trace file
      shutil.copy (os.path.join (configuration_location, ns3['opt']['trace-file']), inputs)
      ns3['opt']['trace-file'] = os.path.join (inputs, os.path.basename (ns3['opt']['trace-file']))
    else:
      # create trace file
      nswitches = None
      with open (ns3['req']['topology-file'], 'r') as topfile:
        data = topfile.read().splitlines ()
        nswitches = int (data[0].split (' ')[1])
      nnodes = ncount + nswitches
      ns3['opt']['trace-file'] = os.path.join (inputs, 'ns3_trace.txt')
      with open (ns3['opt']['trace-file'], 'w') as tracefile:
        tracefile.write ('%d\n' % (nnodes))
        tracefile.write (' '.join ([str(i) for i in range (nnodes)]))

    if ns3['opt']['flow-file']:
      # copy flow file
      shutil.copy (os.path.join (configuration_location, ns3['opt']['flow-file']), inputs)
      ns3['opt']['flow-file'] = os.path.join (inputs, os.path.basename (ns3['opt']['flow-file']))
    else:
      # create flow file
      ns3['opt']['flow-file'] = os.path.join (inputs, 'ns3_flow.txt')
      with open (ns3['opt']['flow-file'], 'w') as flowfile:
        flowfile.write ('\n') # just write an empty file

    # ensure outputs are in place
    if not ns3['opt']['fct-output-file']:
      ns3['opt']['fct-output-file'] = os.path.join (outputs, 'fct.txt')
    if not ns3['opt']['trace-output-file']:
      ns3['opt']['trace-output-file'] = os.path.join (outputs, 'trace.tr')
    if not ns3['opt']['pfc-output-file']:
      ns3['opt']['pfc-output-file'] = os.path.join (outputs, 'pfc.txt')
    if not ns3['opt']['qlen-mon-file']:
      ns3['opt']['qlen-mon-file'] = os.path.join (outputs, 'qlen.txt')

    # create config
    # write topology
    cfg.write ('TOPOLOGY_FILE %s\n' % (ns3['req']['topology-file']))
    # write all other
    for key in ns3['opt'].keys ():
      if isinstance (ns3['opt'][key], list):
        cfg.write ('%s %s\n' % (key.replace ('-', '_').upper (), ' '.join ([str (s) for s in ns3['opt'][key]])))
      else:
        cfg.write ('%s %s\n' % (key.replace ('-', '_').upper (), ns3['opt'][key]))

  # make system
  scfg = os.path.join (inputs, 'system_cfg.yaml')
  with open (scfg, 'w') as cfg:
    cfg.write ('{\n')
    # -- required
    for arg in system['req'].keys ():
      line = ', '.join (['"%s"' % x for x in system['req'][arg]])
      cfg.write ('  "{}-implementation": [ '.format (arg) + line + ' ],\n')
    # -- optional
    for arg in system['opt'].keys ():
      value = '"%s"' % system['opt'][arg] if isinstance (system['opt'][arg], str) else system['opt'][arg]
      cfg.write ('  "{}": '.format (arg) + str (value) + ',\n')
    cfg.write ('  "trace-enabled": 1\n')
    cfg.write ('}\n')

  # make logger (edit this)
  lcfg = os.path.join (inputs, 'log_cfg.toml')
  with open (lcfg, 'w') as cfg:
    cfg.write ('[[sink]]\n')
    cfg.write ('name = "sink"\n')
    cfg.write ('type = "color_stdout_sink_mt"\n')
    cfg.write ('\n')
    cfg.write ('[[logger]]\n')
    cfg.write ('name = "logger"\n')
    cfg.write ('sinks = ["sink"]\n')
    cfg.write ('level = "trace"\n')

  # make memory
  mcfg = os.path.join (inputs, 'memory_cfg.json')
  with open (mcfg, 'w') as cfg:
    cfg.write ('{\n')
    cfg.write ('  "memory-type": "NO_MEMORY_EXPANSION"\n')
    cfg.write ('}\n')

  # make workload
  workload = [w for w in design['workloads'] if w['name'] == target_workload][0]
  wdir = os.path.join (inputs, workload['name'])
  os.makedirs (wdir, exist_ok = True)
  configuration_location = os.path.dirname (os.path.abspath (filename))
  workload_location = os.path.join (configuration_location, workload['location'])
  if workload['format'] == 'text':
    # check if sample text module
    if workload.get ('sample', False) == True:
      # set workload location to appropriate value
      workload_location = os.path.join (
              '/opt/astra-sim/astra-sim-dev/examples/text_converter',
              'text_workloads')
    # get num passes
    npasses = workload['num-passes']
    # convert text to chakra
    subprocess.run (['chakra_converter', 
                     'Text',
                     '--input', os.path.join(workload_location, workload['name'] + '.txt'),
                     '--output', os.path.join(wdir, os.path.splitext (workload['name'])[0]),
                     '--num-npus', str (ncount),
                     '--num-passes', str (npasses) # exit this
                     ])
  elif workload['format'] == 'chakra':
    # copy chakra into project directory (not recommended)
    pattern = workload['name'] + r'\.[0-9]+\.et'
    chakra_workloads = [c for c in os.listdir (workload_location) if re.match(pattern, c)]
    for chakra_workload in chakra_workloads:
      shutil.copy (os.path.join (workload_location, chakra_workload), wdir)
  elif workload['format'] == 'script':
    # run python script to generate
    sys.path.append (workload_location)
    script = importlib.import_module (workload['name'])
    # looks for 'generate_et' function within module that looks like this:
    #   def generate_et (node_index: int, et: io.TextIOWrapper) -> None:
    #     ...
    #     encode (et, <ChakraNode>) # Call at least once to add a node
    #     ...
    for i in range (ncount):
      with open (os.path.join (wdir, workload['name'] + '.%s.et' % (i)), 'wb') as et:
        encode (et, GlobalMetadata (version='0.0.4'))
        script.generate_et (i, et)
  elif workload['format'] == 'yaml':
    # read yaml file with node declarations and generate
    wyml = None
    with open (os.path.join (workload_location, workload['name'] + '.yaml'), 'r') as file:
      wyml = yaml.load (file, Loader = Loader)
    for i in range (ncount):
      with open (os.path.join (wdir, workload['name'] + '.%s.et' % (i)), 'wb') as et:
        encode (et, GlobalMetadata (version = '0.0.4'))
        etgenerate.generate (i, wyml['workload'][i]['nodes'], et)
  else:
    print ('Invalid workload parameter (%s)' % (workload['format']))
    sys.exit (1)

  # make run script
  rscript = os.path.join (proj, 'run.sh')
  with open (rscript, 'w') as script:
    script.write ('#!/bin/bash\n')
    script.write ('\n')
    script.write ('PROJECTPATH=$(dirname "$(realpath $0)")\n')
    script.write ('\n')
    script.write ('rm -rf ${PROJECTPATH}/log\n')
    script.write ('mkdir -p ${PROJECTPATH}/tracker\n')
    script.write ('\n')
    script.write (binary + ' \\\n')
    wcfg = os.path.join (wdir, workload['name'])
    line = '  --workload-configuration="' + wcfg + '" \\\n'
    line = line.replace (str (proj),'${PROJECTPATH}')
    script.write (line)
    line = '  --system-configuration="' + scfg + '" \\\n'
    line = line.replace (str (proj),'${PROJECTPATH}')
    script.write (line)
    line = '  --logical-topology-configuration="' + ncfg + '" \\\n'
    line = line.replace (str (proj),'${PROJECTPATH}')
    script.write (line)
    line = '  --remote-memory-configuration="' + mcfg + '" \\\n'
    line = line.replace (str (proj),'${PROJECTPATH}')
    script.write (line)
    line = '  --comm-group-configuration="\\"empty\\"" \\\n'
    script.write (line)
    line = '  --network-configuration="' + nscfg + '" \\\n'
    line = line.replace (str (proj), '${PROJECTPATH}')
    script.write (line)
    line = '  --logging-configuration="' + lcfg + '" | tee >(grep "\[tracker\]" > tracker/log.log) \n'
    line = line.replace (str (proj),'${PROJECTPATH}')
    script.write (line)
    script.write ('\n')
  os.chmod (rscript, 0o755)

def generate_config_analytical (filename, design, target_design, target_workload, binary, overwrite = True):  
  # network
  network = {
    # -- required arguments
    'req': {
      'topology': [],
      'npus': [],
      'bandwidth': [],
      'latency': []
    },
    # -- optional arguments
    'opt': {
    }
  }

  # system
  system = {
    # -- required arguments
    'req': {
      'all-reduce': [],
      'all-gather': [],
      'reduce-scatter': [],
      'all-to-all': []
    },
    # -- optional arguments
    'opt': {
      'scheduling-policy': 'LIFO',
      'roofline-enabled': 1,
      'replay-only': 0,
      'endpoint-delay': 0,
      'preferred-dataset-splits': 4,
      'active-chunks-per-dimension': 1,
      'collective-optimization': 'localBWAware',
      'local-mem-bw': 50,
      'boost-mode': 0,
      'peak-perf': 0.001, # Measured in TFLOPS
    }
  }

  for dim in design['dims']:
    if dim['valid'] == True:
      # network
      # -- required arguments
      for arg in network['req'].keys ():
        if arg == 'topology':
          network['req'][arg].append (Mappings.Network[dim[arg]])
        else:
          network['req'][arg].append (dim[arg])
      # -- optional arguments
      for arg in network['opt'].keys ():
        if arg in dim.keys ():
          network['opt'][arg] = dim[arg]
      # system
      # -- required arguments
      for arg in system['req'].keys ():
        system['req'][arg].append (Mappings.System[dim[arg]])
      # -- optional argumens
      for arg in system['opt'].keys ():
        if arg in dim.keys ():
          system['opt'][arg] = dim[arg]
  # get npu count
  ncount = 1
  for nc in network['req']['npus']:
    ncount = ncount * nc

  # create project dir
  cwd = os.getcwd ()
  proj = os.path.join (cwd, target_design + '-' + target_workload)
  if os.path.isdir (proj) and overwrite:
    shutil.rmtree (proj)
  os.makedirs (proj, exist_ok = True)

  # make inputs
  inputs = os.path.join (proj, 'inputs')
  os.makedirs (inputs, exist_ok = True)

  # make network
  ncfg = os.path.join (inputs, 'network_cfg.yaml')
  with open (ncfg, 'w') as cfg:
    # -- required
    for arg in network['req'].keys ():
      line = ', '.join ([str (i) for i in network['req'][arg]])
      if arg == 'npus':
        cfg.write ('npus_count: [ ' + line + ' ]\n')
      else:
        cfg.write (arg + ': [ ' + line + ' ]\n')
    # -- optional
    for arg in network['opt'].keys ():
      cfg.write (arg + ': ' + str (network['opt'].keys ()))

  # make system
  scfg = os.path.join (inputs, 'system_cfg.yaml')
  with open (scfg, 'w') as cfg:
    cfg.write ('{\n')
    # -- required
    for arg in system['req'].keys ():
      line = ', '.join (['"%s"' % x for x in system['req'][arg]])
      cfg.write ('  "{}-implementation": [ '.format (arg) + line + ' ],\n')
    # -- optional
    for arg in system['opt'].keys ():
      value = '"%s"' % system['opt'][arg] if isinstance (system['opt'][arg], str) else system['opt'][arg]
      cfg.write ('  "{}": '.format (arg) + str (value) + ',\n')
    cfg.write ('  "trace-enabled": 1\n')
    cfg.write ('}\n')

  # make logger (edit this)
  lcfg = os.path.join (inputs, 'log_cfg.toml')
  with open (lcfg, 'w') as cfg:
    cfg.write ('[[sink]]\n')
    cfg.write ('name = "sink"\n')
    cfg.write ('type = "color_stdout_sink_mt"\n')
    cfg.write ('\n')
    cfg.write ('[[logger]]\n')
    cfg.write ('name = "logger"\n')
    cfg.write ('sinks = ["sink"]\n')
    cfg.write ('level = "trace"\n')

  # make memory
  mcfg = os.path.join (inputs, 'memory_cfg.json')
  with open (mcfg, 'w') as cfg:
    cfg.write ('{\n')
    cfg.write ('  "memory-type": "NO_MEMORY_EXPANSION"\n')
    cfg.write ('}\n')

  # make workload
  workload = [w for w in design['workloads'] if w['name'] == target_workload][0]
  wdir = os.path.join (inputs, os.path.splitext (workload['name'])[0])
  os.makedirs (wdir, exist_ok = True)
  configuration_location = os.path.dirname (os.path.abspath (filename))
  workload_location = os.path.join (configuration_location, workload['location'])
  if workload['format'] == 'text':
    # check if sample text module
    if workload.get ('sample', False) == True:
      # set workload location to appropriate value
      workload_location = os.path.join (
              '/opt/astra-sim/astra-sim-dev/examples/',
              'text_converter')
    # get num passes
    npasses = workload['num-passes']
    # convert text to chakra
    subprocess.run (['chakra_converter', 
                     'Text',
                     '--input', os.path.join(workload_location, workload['name'] + '.txt'),
                     '--output', os.path.join(wdir, os.path.splitext (workload['name'] + '.txt')[0]),
                     '--num-npus', str (ncount),
                     '--num-passes', str (npasses) # exit this
                     ])
  elif workload['format'] == 'chakra':
    # copy chakra into project directory (not recommended)
    pattern = workload['name'] + r'\.[0-9]+\.et'
    chakra_workloads = [c for c in os.listdir (workload_location) if re.match(pattern, c)]
    for chakra_workload in chakra_workloads:
      shutil.copy (os.path.join (workload_location, chakra_workload), wdir)
  elif workload['format'] == 'script':
    # run python script to generate
    sys.path.append (workload_location)
    script = importlib.import_module (workload['name'])
    # looks for 'generate_et' function within module that looks like this:
    #   def generate_et (node_index: int, et: io.TextIOWrapper) -> None:
    #     ...
    #     encode (et, <ChakraNode>) # Call at least once to add a node
    #     ...
    for i in range (ncount):
      with open (os.path.join (wdir, os.path.splitext (workload['name'])[0] + '.%s.et' % (i)), 'wb') as et:
        encode (et, GlobalMetadata (version='0.0.4'))
        script.generate_et (i, et)
  elif workload['format'] == 'yaml':
    # read yaml file with node declarations and generate
    wyml = None
    with open (os.path.join (workload_location, workload['name'] + '.yaml'), 'r') as file:
      wyml = yaml.load (file, Loader = Loader)
    for i in range (ncount):
      with open (os.path.join (wdir, os.path.splitext (workload['name'])[0] + '.%s.et' % (i)), 'wb') as et:
        encode (et, GlobalMetadata (version = '0.0.4'))
        etgenerate.generate (i, wyml['workload'][i], et)
  else:
    print ('Invalid workload parameter (%s)' % (workload['format']))
    sys.exit (1)

  # make run script
  rscript = os.path.join (proj, 'run.sh')
  with open (rscript, 'w') as script:
    script.write ('#!/bin/bash\n')
    script.write ('\n')
    script.write ('PROJECTPATH=$(dirname "$(realpath $0)")\n')
    script.write ('\n')
    script.write ('rm -rf ${PROJECTPATH}/log\n')
    script.write ('mkdir -p ${PROJECTPATH}/tracker\n')
    script.write ('\n')
    script.write (binary + ' \\\n')
    wcfg = os.path.join (wdir, workload['name'])
    line = '  --workload-configuration="' + wcfg + '" \\\n'
    line = line.replace (str (proj),'${PROJECTPATH}')
    script.write (line)
    line = '  --system-configuration="' + scfg + '" \\\n'
    line = line.replace (str (proj),'${PROJECTPATH}')
    script.write (line)
    line = '  --network-configuration="' + ncfg + '" \\\n'
    line = line.replace (str (proj),'${PROJECTPATH}')
    script.write (line)
    line = '  --remote-memory-configuration="' + mcfg + '" \\\n'
    line = line.replace (str (proj),'${PROJECTPATH}')
    script.write (line)
    line = '  --logging-configuration="' + lcfg + '" | tee >(grep "\[tracker\]" > tracker/log.log) \n'
    line = line.replace (str (proj),'${PROJECTPATH}')
    script.write (line)
    script.write ('\n')
  os.chmod (rscript, 0o755)

def generate_config (filename, target_design = None, target_workload = None, overwrite = True):
  # read configuration
  yml = None
  with open (filename, 'r') as file:
    yml = yaml.load (file, Loader = Loader)
  if target_design not in yml.keys ():
    print ('Could not find (%s) in (%s)' % (target_design, filename))
  design = yml[target_design]

  # select flavor
  flavor, tag = design['type'].split (':')
  binary = 'astra-sim'
  if flavor == 'analytical':
    pass
  elif flavor == 'ns3':
    binary = binary + '-ns3'
  else:
    print ('Invalid type give (%s)' % flavor)
    sys.exit (1)

  if tag == 'base':
    pass
  elif tag == 'dev':
    binary = binary + '-dev'
  else:
    print ('Invalid tag give (%s)' % tag)
    sys.exit (1)

  if flavor == 'analytical':
    return generate_config_analytical (filename, design, target_design, target_workload, binary, overwrite)
  elif flavor == 'ns3':
    return generate_config_ns3 (filename, design, target_design, target_workload, binary, overwrite)

def main ():
  parser = argparse.ArgumentParser (
      prog='configure-astra',
      description='Creates astra-sim project directory',
      epilog='')
  parser.add_argument ('-c','--config',default='configuration.yaml')
  parser.add_argument ('-d','--design', required=True)
  parser.add_argument ('-w','--workload', required=True)
  args = parser.parse_args ()
  generate_config (args.config, args.design, args.workload)

if __name__ == '__main__':
  main ()
