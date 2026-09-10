#!/bin/bash
cd "$HOME/projects/srtp"
stdbuf -oL -eL .venv/bin/python src/experiment_2c_stability_ecg5000.py > logs/stability_ecg5000.log 2>&1
echo "=== exit: $? ===" >> logs/stability_ecg5000.log
