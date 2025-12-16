@echo off
REM demo_sweep.cmd
REM One-command demo for Windows CMD

set POMDP_N_WORKERS=4
python scripts\demo_sweep.py --mode AB --device cpu --mc-episodes 30000 --eval-initial 10000 --n-sims 30 --out-root results
