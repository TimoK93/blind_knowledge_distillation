#!/bin/bash

CONDA_ENV=otsu_split

source /home/ehmannlu/tmp/Anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
python -V

echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
python --version
echo "Python environment: $PYTHON_ENV_NAME"

echo "Start python"

DATASET=cifar10
for SEED in 0 1 2 3 4
do
  for NOISE_TYPE in aggre worst rand1 rand2 rand3
  do
    echo $SEED $NOISE_TYPE $DATASET
    python blind_knowledge_dist_training.py --dataset ${DATASET} --noise_type ${NOISE_TYPE} --seed ${SEED} > results/${NOISE_TYPE}_seed_${SEED}/training_log.log
    python learning.py --dataset ${DATASET} --noise_type ${NOISE_TYPE} --seed ${SEED} > results/${NOISE_TYPE}_seed_${SEED}/learning_log.log
    python detection.py --dataset ${DATASET} --noise_type ${NOISE_TYPE} --seed ${SEED} > results/${NOISE_TYPE}_seed_${SEED}/detection_log.log
  done
done

DATASET=cifar100
NOISE_TYPE=noisy100
for SEED in 0 1 2 3 4
do
  echo $SEED $NOISE_TYPE $DATASET
  python blind_knowledge_dist_training.py --dataset ${DATASET} --noise_type ${NOISE_TYPE} --seed ${SEED} > results/${NOISE_TYPE}_seed_${SEED}/training_log.log
  python learning.py --dataset ${DATASET} --noise_type ${NOISE_TYPE} --seed ${SEED} > results/${NOISE_TYPE}_seed_${SEED}/learning_log.log
  python detection.py --dataset ${DATASET} --noise_type ${NOISE_TYPE} --seed ${SEED} > results/${NOISE_TYPE}_seed_${SEED}/detection_log.log
done

echo 'done'
