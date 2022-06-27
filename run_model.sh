#!/bin/bash
#SBATCH --job-name=otsu_split
#SBATCH --mail-user=ehmannlu@tnt.uni-hannover.de
#SBATCH --mail-type=NONE                # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --output=/home/ehmannlu/tmp/TNT_slurm_logs/code_test/%j-train.txt      # Logdatei für den merged STDOUT/STDERR output (%j wird durch slurm job-ID ersetzt)
#SBATCH --partition=gpu_normal_stud  # Partition auf der gerechnet werden soll (Bei GPU Jobs unbedingt notwendig)
#SBATCH --cpus-per-task=8             # Reservierung von zwei Threads
#SBATCH --mem=20G                       # Reservierung von 1 GB RAM Speicher pro Knoten
#SBATCH --gres=gpu:1                  # Reservierung von einer GPU. Es kann ein bestimmter Typ angefordert werden:
#SBATCH --time=24:00:00             # Maximale Laufzeit des Jobs, bis Slurm diesen abbricht (HH:MM:SS)
#SBATCH --array=0-1


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
DATASET=cifar10


for SEED in 24 # 1 2 3 4
do
  for NOISE_TYPE in aggre worst # rand1 rand2 rand3
  do
    if [ $ITER = $SLURM_ARRAY_TASK_ID ]
    then
      echo $SEED $NOISE_TYPE $DATASET
      # mkdir results
      mkdir results/${DATASET}_${NOISE_TYPE}_seed_${SEED}
      python blind_knowledge_dist_training.py --dataset ${DATASET} --noise_type ${NOISE_TYPE} --seed ${SEED} > results/${DATASET}_${NOISE_TYPE}_seed_${SEED}/training_log.log
      # python blind_knowledge_dist_training.py --dataset ${DATASET} --noise_type ${NOISE_TYPE} --seed ${SEED} > results/${NOISE_TYPE}_seed_${SEED}_alpha1__${ALPHA1}_training_log.log
      #python learning.py --dataset ${DATASET} --noise_type ${NOISE_TYPE} --seed ${SEED} > results/${NOISE_TYPE}_seed_${SEED}_learning_log.log
      #python detection.py --dataset ${DATASET} --noise_type ${NOISE_TYPE} --seed ${SEED} > results/${NOISE_TYPE}_seed_${SEED}_detection_log.log
    fi
    ITER=$(($ITER+1))
  done
done

echo 'done'
