#!/bin/zsh

# 1. Install Poetry if not already installed
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> $HOME/.bashrc
    source $HOME/.bashrc
fi

# 2. Initialize a new Poetry project if 'pyproject.toml' is not present
if [ ! -f "pyproject.toml" ]; then
    echo "Creating a new Poetry project..."
    poetry new project_name
    cd project_name  # Replace 'project_name' with your actual project name
fi

# 3. Ensure the required Python version is set in 'pyproject.toml'
echo "Make sure that the required Python version is set in your 'pyproject.toml' file."

# 4. Create and activate the Poetry virtual environment
poetry install

# 5. Enter the virtual environment shell
poetry shell

# 6. Continue with your project's setup
echo "Continue with your project's setup."
