#!/usr/bin/bash

# Install python requirements
pip install -r python/requirements.txt

# Install Mojo
pip install mojo

# Instantiate julia depot
julia --project=julia/ -E "using Pkg; Pkg.instantiate()"
