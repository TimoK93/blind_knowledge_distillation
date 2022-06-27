SEED=0

mkdir results
mkdir results/cifar10_aggre_seed_${SEED}
mkdir results/cifar10_worst_seed_${SEED}
mkdir results/cifar10_rand1_seed_${SEED}

python blind_knowledge_dist_training.py --dataset cifar10 --noise_type aggre --seed ${SEED} > results/cifar10_aggre_seed_${SEED}/training.log &

python blind_knowledge_dist_training.py --dataset cifar10 --noise_type worst --seed ${SEED} > results/cifar10_worst_seed_${SEED}/training.log &

python blind_knowledge_dist_training.py --dataset cifar10 --noise_type rand1 --seed ${SEED} > results/cifar10_rand1_seed_${SEED}/training.log
