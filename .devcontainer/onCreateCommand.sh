#!/usr/bin/bash

# Create venv
python3 -m venv .venv

source .venv/bin/activate

# Install python requirements
pip install -r python/requirements.txt

# Install Mojo
pip install mojo

# Instantiate julia depot
julia -E "using Pkg; Pkg.add(\"StaticArrays\")"
