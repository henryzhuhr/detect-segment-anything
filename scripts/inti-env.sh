# CUSTOM_PYTHON_VERSION=3.10 # Uncomment and set to the desired Python version

bash scripts/get-resource.sh
# if [ ! -d downloads/CLIP ]; then
#     echo "Run 'bash scripts/get-resource.sh' first to download resource."
#     exit 1
# fi

PROJECT_HOME=$(pwd)
ENV_PATH=./.env

if [ ! -d $ENV_PATH ]; then
    SYS_PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
    if [ -z "$CUSTOM_PYTHON_VERSION" ]; then
        ENV_PYTHON_VERSION=$SYS_PYTHON_VERSION
    else
        ENV_PYTHON_VERSION=$CUSTOM_PYTHON_VERSION
    fi
    conda create -p $ENV_PATH -y python=$ENV_PYTHON_VERSION 
else
    echo "Conda environment '$ENV_PATH' already exists."
fi

eval "$(conda shell.bash hook)"
conda activate $ENV_PATH
echo "Activated $(python --version) in ($ENV_PATH)"

# install pytorch according to the CUDA version
if [ ! -z "${CUDA_VERSION}" ]; then
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | tr -d '.')
else
    python3 -m pip install torch torchvision
    echo "CUDA_VERSION is not set. Installing CPU version of PyTorch."
fi

python3 -m pip install -r requirements.txt

cd $PROJECT_HOME/.cache/repo/GroundingDINO
python3 -m pip install -e .
# python3 -m pip install -e downloads/CLIP