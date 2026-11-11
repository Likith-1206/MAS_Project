#!/usr/bin/env bash
# Animated simulations on all five cities, every model side by side:
#   RAPID-Light final, RAPID-Light multi-city, Independent DQN, MAPPO, fixed-time
# Result: viz/<city>... .html per city + viz/index.html (opened automatically: pick any city there).
# RAPID-Light needs results/nyc16/rapid_v2_mixed/rapid2_mixedtrained_s0.pt and results/multicity/rapid_multicity_s0.pt.
# DQN / MAPPO models missing in models/<city>/ are trained on that city first (the slow part).
# Options are passed on, e.g.  ./simulate_all.sh --average      (12-test-episode averages + representative replay)
# Fewer models:  CTRL="rapid" ./simulate_all.sh
set -e
cd "$(dirname "$0")"
CTRL=${CTRL:-"rapid rapid_multicity independent mappo"}
python src/simulate.py maps/nyc16 maps/pune_test maps/mumbai maps/bangalore maps/jersey \
  --controllers $CTRL --compare fixed_program --open "$@"