#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=tesi_pcarboni
#SBATCH --job-name=cl_detr_param_test
#SBATCH --nodes=1
#SBATCH --array=0%100
#SBATCH --output="./out/CL-DETR_%A_%a.out"
#SBATCH --error="./err/CL-DETR_%A_%a.err"
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_RTX5000_16G"

echo "Testing cl_detr"
echo "Running on Hostname: $(hostname)"
nvidia-smi

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate tesi
echo "Active Python Environment:"
conda env list | grep '*' # This shows the active Conda environment
python --version
echo "Python Path:"
which python

srun python -u -m site

# Dynamic Port Assignment to Avoid Conflicts
MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID + 150))
export MASTER_PORT

# wandb offline
#export WANDB__SERVICE_WAIT=300
#export PYTHONPATH=/work/tesi_pcarboni/cl_detr
cd /work/tesi_pcarboni/cl_detr
cd ./models/ops
sh ./make.sh

arguments=(
    "--nproc_per_node 4 /work/tesi_pcarboni/CL-DETR/main.py"
    #"/work/tesi_pcarboni/cl_detr/main.py --batch_size 4 --epochs 5 --lr 1e-2 --num_queries 50 --enc_layers 4 --dec_layers 4 --output_dir /work/tesi_pcarboni/results/exp2"
    #"/work/tesi_pcarboni/cl_detr/main.py --batch_size 4 --epochs 5 --lr 1e-3 --num_queries 50  --enc_layers 4 --dec_layers 4 --output_dir /work/tesi_pcarboni/results/exp3"
)


sleep $(($RANDOM % 20)); srun python -u tools/launch.py ${arguments[$SLURM_ARRAY_TASK_ID]}
