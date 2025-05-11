#!/bin/bash
#SBATCH --job-name=finetune_SaulLM
#SBATCH --output=logs/Saul_FT.out
#SBATCH --error=logs/Saul_FT.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

echo "=== Starting job on $(hostname) at $(date) ==="

# Activate environment
source /home/njuttu_umass_edu/venvs/torch311_env/bin/activate

# Ensure venv paths are active
export PYTHONPATH="/home/njuttu_umass_edu/venvs/torch311_env/lib/python3.11/site-packages"
export PATH="/home/njuttu_umass_edu/venvs/torch311_env/bin:$PATH"

# Optional: log versions
echo "Python version: $(python --version)"
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"

# Move into script directory
cd /home/njuttu_umass_edu/685/ZeroShotAnomolyDetection-1/SaulLM:7B

echo "Running fine-tuning..."
python run_finetune.py

echo "=== Job finished at $(date) ==="