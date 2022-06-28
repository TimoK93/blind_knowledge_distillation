SEED=0

mkdir results
mkdir results/cifar10_aggre_seed_${SEED}
mkdir results/cifar10_worst_seed_${SEED}
mkdir results/cifar10_rand1_seed_${SEED}

# training - Please modify with CUDA_VISIBLE_DEVICES and multiprocessing if you have the resources
nohup python blind_knowledge_dist_training.py --dataset cifar10 --noise_type aggre --seed ${SEED} > results/cifar10_aggre_seed_${SEED}/training.log &
nohup python blind_knowledge_dist_training.py --dataset cifar10 --noise_type worst --seed ${SEED} > results/cifar10_worst_seed_${SEED}/training.log &
nohup python blind_knowledge_dist_training.py --dataset cifar10 --noise_type rand1 --seed ${SEED} > results/cifar10_rand1_seed_${SEED}/training.log

# eval test_acc with learning.py
python learning.py --dataset cifar10 --noise_type aggre --seed ${SEED} > results/cifar10_aggre_seed_${SEED}/learning.log
python learning.py --dataset cifar10 --noise_type worst --seed ${SEED} > results/cifar10_worst_seed_${SEED}/learning.log
python learning.py --dataset cifar10 --noise_type rand1 --seed ${SEED} > results/cifar10_rand1_seed_${SEED}/learning.log

# eval detection-metrics with detection.py
python detection.py --dataset cifar10 --noise_type aggre --seed ${SEED} > results/cifar10_aggre_seed_${SEED}/detection.log
python detection.py --dataset cifar10 --noise_type worst --seed ${SEED} > results/cifar10_worst_seed_${SEED}/detection.log
python detection.py --dataset cifar10 --noise_type rand1 --seed ${SEED} > results/cifar10_rand1_seed_${SEED}/detection.log