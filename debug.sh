#!/bin/bash
#SBATCH -p gpu2


WORKDIR="/homes/hnakayama/realtime"
cd $WORKDIR

uv run python debug.py