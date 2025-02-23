#! /bin/bash

set -e

# setting up the server
echo "THIS IS MEANT TO BE RUN ON A NEW SERVER. DON'T RUN IT ON YOUR LOCAL MACHINE."
echo "ARE YOU SURE YOU WANT TO CONTINUE? (y/n)"
read -r answer
if [ "$answer" != "y" ]; then
    echo "exiting"
    exit 1
fi

USE_CONDA=false

if [ "$USE_CONDA" = true ]; then
    if ! command -v conda &> /dev/null; then
        # Conda not found, proceed with installation

        echo "installing miniconda"
        CONDA_PATH="/workspace/miniconda3"

        mkdir -p $CONDA_PATH
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $CONDA_PATH/miniconda.sh
        bash $CONDA_PATH/miniconda.sh -b -u -p $CONDA_PATH
        rm $CONDA_PATH/miniconda.sh
        $CONDA_PATH/bin/conda init bash
        source ~/.bashrc # just to make sure the path is set and conda is available


        conda create -n ml python=3.10
    fi

    echo "activating conda environment"
    conda activate ml
fi

echo "installing pipx"
apt update

echo "installing apt dependencies"
apt install tmux
apt install htop
apt install nvtop
apt install nano

echo "installing poetry"
pipx install poetry
pipx ensurepath
source ~/.bashrc # just to make sure the path is set

echo "installing dependencies with poetry"
# Configure poetry to not create virtual environment and use conda instead
# poetry config virtualenvs.create false
# poetry config virtualenvs.in-project false
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true
poetry install


echo "Setup complete!"
