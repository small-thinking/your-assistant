#!/bin/bash

# 1. Download and install Miniconda if not already installed
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
    fi
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> $HOME/.bashrc
    source $HOME/.bashrc
    conda init
    source $HOME/.bashrc
fi

# 2. Create and activate the 'assist' virtual environment
conda create -n assist python=3.9 -y
conda activate assist

# 3. Install Poetry if not already installed
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python -
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> $HOME/.bashrc
    source $HOME/.bashrc
fi

# 4. Activate the Poetry environment (implicitly activated when using 'poetry' commands)

# 5. Install dependencies using Poetry
poetry install
