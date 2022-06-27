SEED=0

python blind_knowledge_dist_training.py --dataset cifar10 --noise_type aggre --seed ${SEED} > cifar10_aggre_seed_${SEED}_training.log &

python blind_knowledge_dist_training.py --dataset cifar10 --noise_type worst --seed ${SEED} > cifar10_worst_seed_${SEED}_training.log &

python blind_knowledge_dist_training.py --dataset cifar10 --noise_type rand1 --seed ${SEED} > cifar10_rand1_seed_${SEED}_training.log &
