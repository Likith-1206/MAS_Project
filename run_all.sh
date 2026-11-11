#!/bin/bash
# Pair B: all 15 training runs (5 methods x 3 seeds x 120 episodes) + baselines.
# Split between laptops with e.g.:  SEEDS="0" ./run_all.sh   /   SEEDS="1 2" ./run_all.sh
cd "$(dirname "$0")"
EP=${EPISODES:-120}
SEEDS=${SEEDS:-"0 1 2"}
METHODS_LIST=${METHODS_LIST:-"independent coop shared_c0 shared_c1 shared_c2"}
if [ -z "$SKIP_BASELINES" ]; then python3 src/run_experiments.py baselines; fi
for s in $SEEDS; do
  for m in $METHODS_LIST; do python3 src/run_experiments.py train --method $m --episodes $EP --seed $s; done
done
echo ALL_DONE