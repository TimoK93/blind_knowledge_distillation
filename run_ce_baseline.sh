#!/bin/bash
#SBATCH --job-name=code_test
#SBATCH --mail-user=ehmannlu@tnt.uni-hannover.de
#SBATCH --mail-type=NONE                # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --output=/home/ehmannlu/tmp/TNT_slurm_logs/code_test/%j-train.txt      # Logdatei für den merged STDOUT/STDERR output (%j wird durch slurm job-ID ersetzt)
#SBATCH --partition=gpu_normal_stud  # Partition auf der gerechnet werden soll (Bei GPU Jobs unbedingt notwendig)
#SBATCH --cpus-per-task=8             # Reservierung von zwei Threads
#SBATCH --mem=20G                       # Reservierung von 1 GB RAM Speicher pro Knoten
#SBATCH --gres=gpu:1                  # Reservierung von einer GPU. Es kann ein bestimmter Typ angefordert werden:
#SBATCH --time=24:00:00             # Maximale Laufzeit des Jobs, bis Slurm diesen abbricht (HH:MM:SS)
#SBATCH --array=0-3

export NAME=ehmannlu
# export CFG_FILE=config/lukas/new_correction_training.yaml

# cd ..

HOSTNAME=$(srun hostname)
CONDA_ENV=otsu_split

source /home/${NAME}/tmp/Anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
python -V

echo "Name: $NAME"
echo "Hostname: $HOSTNAME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "GPU IDs: $SLURM_JOB_GPUS"
python --version
echo "Python environment: $PYTHON_ENV_NAME"
echo "PATH: $PATH"

echo "Start python"

ITER=0

#for SEED in 12
#do
#  if [ $ITER = $SLURM_ARRAY_TASK_ID ]
#  then
#    echo $ITER $CFG_FILE $SEED $NOISE_TYPE $SPLIT_DATASET
#    python main.py $CFG_FILE --arguments seed ${SEED} data.train.noise_type ${NOISE_TYPE} training.split_dataset ${SPLIT_DATASET}  # training.constraints.only_positive_logits ${ONLY_POSITIVE_LOGITS} training.constraints.deactivate_gt_prediction ${DEACTIVATE_GT_PREDICTION}
#    #python main.py $CFG_FILE --arguments seed ${SEED} data.train.noise_type ${NOISE_TYPE} training.alpha_post_split.trusted_to_0 ${TRUSTED_TO_0} training.alpha_post_split.untrusted_to_1 ${UNTRUSTED_TO_1} &
#    #python main.py $CFG_FILE --arguments seed "1"${SEED} data.train.noise_type ${NOISE_TYPE} training.alpha_post_split.trusted_to_0 ${TRUSTED_TO_0} training.alpha_post_split.untrusted_to_1 ${UNTRUSTED_TO_1}
#  fi
#  ITER=$(($ITER+1))
#done

if [ $ITER = $SLURM_ARRAY_TASK_ID ]
then
  python ce_baseline.py --dataset cifar10 --noise_type worst --val_ratio 0.1 > logs/c10_worst.log
fi

ITER=$(($ITER+1))

if [ $ITER = $SLURM_ARRAY_TASK_ID ]
then
  python ce_baseline.py --dataset cifar10 --noise_type rand1 --val_ratio 0.1 > logs/c10_rand1.log
fi

ITER=$(($ITER+1))

if [ $ITER = $SLURM_ARRAY_TASK_ID ]
then
  python ce_baseline.py --dataset cifar10 --noise_type aggre --val_ratio 0.1 > logs/c10_aggre.log
fi

ITER=$(($ITER+1))

if [ $ITER = $SLURM_ARRAY_TASK_ID ]
then
  python ce_baseline.py --dataset cifar100 --noise_type noisy100 --val_ratio 0.1 > logs/c100.log
fi

echo 'done'
