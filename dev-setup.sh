#!/bin/zsh

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
env_name="assist"
if [ "" = $(conda env list | awk '$0=$1' | grep -w "$env_name") ]; then
    echo "Conda environment '$env_name' does not exists, create one."
    command="conda create --name ${env_name} python=3.9 -y"
    echo "Execute command " $command
    eval $command
fi
echo "Activate conda environment '$env_name'."
command="source activate $env_name"
eval command

# 3. Install Poetry if not already installed
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> $HOME/.bashrc
    source $HOME/.bashrc
fi

# 4. Activate the Poetry environment (implicitly activated when using 'poetry' commands)
command="poetry shell"
echo $command
eval $command

# 5. Install dependencies using Poetry
Manually run "poetry install"
