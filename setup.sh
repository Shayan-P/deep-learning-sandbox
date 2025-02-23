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

echo "installing miniconda"
CONDA_PATH="/workdir/miniconda3"

mkdir -p $CONDA_PATH
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $CONDA_PATH/miniconda.sh
bash $CONDA_PATH/miniconda.sh -b -u -p $CONDA_PATH
rm $CONDA_PATH/miniconda.sh
source ~/.bashrc # just to make sure the path is set and conda is available

echo "activating conda environment"
conda create -n ml python=3.10
conda activate ml

echo "installing pipx"
apt update

echo "installing poetry"
pipx install poetry
pipx ensurepath
source ~/.bashrc # just to make sure the path is set

echo "installing dependencies with poetry"
# Configure poetry to not create virtual environment and use conda instead
poetry config virtualenvs.create false
poetry config virtualenvs.in-project false
poetry install
