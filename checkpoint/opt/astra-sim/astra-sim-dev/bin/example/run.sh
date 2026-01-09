#!/bin/bash

rm -rf $(dirname "$(realpath $0)")/log
./astra-sim-dev \
  --workload-configuration=./inputs/backup/backup \
  --system-configuration=./inputs/system_cfg.yaml \
  --network-configuration=./inputs/network_cfg.yaml \
  --remote-memory-configuration=./inputs/memory_cfg.json \
  --logging-configuration=./inputs/log_cfg.toml \

