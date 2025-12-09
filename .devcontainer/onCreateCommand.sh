#!/usr/bin/bash

# Create venv
python3 -m venv .venv

source .venv/bin/activate

# Install python requirements
pip install -r python/requirements.txt

# Install Mojo
pip install mojo==0.25.7.0

# Instantiate julia depot
julia --project=julia/ -E "using Pkg; Pkg.instantiate()"
