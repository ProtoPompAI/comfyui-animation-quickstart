# References
# https://waylonwalker.com/install-miniconda/
# https://stackoverflow.com/questions/76892668/shell-script-check-if-the-right-conda-environment-active-or-does-it-exist

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONDA_ENV_LOC="${SCRIPT_DIR}/installer_files/env"

# Install Miniconda to installer_files if an Anaconda distribution is not found on Path
if ! [[ $(command -v conda)  ]]; then
  export PATH="${SCRIPT_DIR}/installer_files/conda/bin:$PATH"
  if ! [[ $(command -v conda)  ]]; then
    (echo "Installing Miniconda to: ${SCRIPT_DIR}/installer_files/conda")
    mkdir -p ${SCRIPT_DIR}/installer_files/
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${SCRIPT_DIR}/installer_files/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -p ${SCRIPT_DIR}/installer_files/conda
    # export PATH="$HOME/miniconda/bin:$PATH"
    # source "$HOME/miniconda/bin/activate"
    rm -rf ${SCRIPT_DIR}/installer_files/miniconda.sh
  fi
fi

# Install a conda environment if it does not exist
if ! conda env list | grep -q "\b${CONDA_ENV_LOC}\b"; then
  (echo "Installing Miniconda to: $CONDA_ENV_LOC")
  conda create -p $CONDA_ENV_LOC python=3.11 -y
fi

# Running ComfyUI setup script
$CONDA_ENV_LOC/bin/python comfyui_downloads.py
