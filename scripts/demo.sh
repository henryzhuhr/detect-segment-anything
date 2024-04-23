ENV_PATH="./.env"
eval "$(conda shell.bash hook)"
conda activate $ENV_PATH
echo "Activated $(python --version) in ($ENV_PATH)"

python3 main.py
