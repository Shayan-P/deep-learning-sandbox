#! /bin/bash

# Function to execute command and handle errors
run_cmd() {
    if ! "$@"; then
        echo "Error occurred while running: $*"
        return 1
    fi
}

# setting up the server
echo "THIS IS MEANT TO BE RUN ON A NEW SERVER. DON'T RUN IT ON YOUR LOCAL MACHINE."
echo "ARE YOU SURE YOU WANT TO CONTINUE? (y/n)"
read -r answer
if [ "$answer" != "y" ]; then
    echo "exiting"
    return 1
fi

USE_CONDA=false

if [ "$USE_CONDA" = true ]; then
    if ! command -v conda &> /dev/null; then
        # Conda not found, proceed with installation
        echo "installing miniconda"
        CONDA_PATH="/workspace/miniconda3"

        run_cmd mkdir -p $CONDA_PATH
        run_cmd wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $CONDA_PATH/miniconda.sh
        run_cmd bash $CONDA_PATH/miniconda.sh -b -u -p $CONDA_PATH
        run_cmd rm $CONDA_PATH/miniconda.sh
        run_cmd $CONDA_PATH/bin/conda init bash
        run_cmd source ~/.bashrc # just to make sure the path is set and conda is available

        run_cmd conda create -n ml python=3.10
    fi

    echo "activating conda environment"
    run_cmd conda activate ml
fi

run_cmd apt update

echo "installing pipx"
run_cmd apt install pipx

echo "installing apt dependencies"
run_cmd apt install tmux
run_cmd apt install htop
run_cmd apt install nvtop
run_cmd apt install nano

echo "installing poetry"
run_cmd pipx install poetry
run_cmd pipx ensurepath
run_cmd source ~/.bashrc # just to make sure the path is set

echo "installing dependencies with poetry"
run_cmd poetry config virtualenvs.create true
run_cmd poetry config virtualenvs.in-project true
run_cmd poetry install

echo "Setup complete!"
return 0
