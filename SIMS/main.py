#!/usr/bin/env python
"""
Main script to run all simulation scripts.

This script will sequentially run the following simulation scripts from the SIMS folder:
  - wavephasefocsweep.py
  - wavephasefocused.py
  - wavephasesync.py
  - wavephasesyncsweep.py
  - wavephasestanding.py
  - wavephasestandingfocsweep.py
  - wavephasestandingsyncsweep.py

Each simulation script computes a simulation scenario and saves its resulting figure
to the "exort" folder.
"""

import os
import subprocess
import sys

def run_simulation_scripts():
    # List of simulation scripts to run (assumed relative to the directory of main.py).
    scripts = [
        "wavephasefocsweep.py",
        "wavephasefocused.py",
        "wavephasesync.py",
        "wavephasesyncsweep.py",
        "wavephasestandingfoc.py",
        "wavephasestandingfocsweep.py",
        "wavephasestandingsyncsweep.py"
    ]
    
    for script in scripts:
        # Check if the simulation script exists.
        if not os.path.exists(script):
            print(f"Script not found: {script}", file=sys.stderr)
            continue
        
        print(f"Running {script}...")
        try:
            # Run the simulation script using Python.
            subprocess.run(["python", script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}", file=sys.stderr)

if __name__ == '__main__':
    run_simulation_scripts() 