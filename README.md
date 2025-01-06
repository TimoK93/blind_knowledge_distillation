# Blind Knowledge Distillation

---
<p align="center">
  <img src="./img/OverallFrameworkV5.png" />
</p>



This repository is the official Pytorch implementation of [Blind Knowledge Distillation](https://arxiv.org/abs/2211.11355) which is competing 
on the Noisy Labels Challenge (ICLR2022) [http://www.noisylabels.com/](http://www.noisylabels.com/). This work
aims to enhance the knowledge about robust learning with noisy labels and detect corrupted training labels. 
It is evaluated on the recently published CIFAR-N dataset 
([Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations](https://openreview.net/forum?id=TBWA6PLJZQm&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2022%2FConference%2FAuthors%23your-submissions))).

This code is based on the [competitions baseline code](https://github.com/UCSC-REAL/cifar-10-100n/tree/ijcai-lmnl-2022).  
## Installation

---
We recommend using conda and python 3.8 to install and run our framework. To install the environment follow the next code
lines:
```
conda create -n distillation python=3.8
conda activate distillation
conda install -c nvidia/label/cuda-11.3.1 cuda-toolkit 
python -m pip install -r requirements.txt
```

## Quick start

---
To train and evaluate our framework, run
````shell
export DATASET=cifar10
export NOISE_TYPE=aggre
export SEED=42
mkdir results
mkdir results/${DATASET}_${NOISE_TYPE}_seed_${SEED}
python blind_knowledge_dist_training.py --dataset ${DATASET} --noise_type ${NOISE_TYPE} --seed ${SEED} > results/${DATASET}_${NOISE_TYPE}_seed_${SEED}/training.log
python learning.py --dataset ${DATASET} --noise_type ${NOISE_TYPE} --seed ${SEED} > results/${DATASET}_${NOISE_TYPE}_seed_${SEED}/learning.log
python detection.py --dataset ${DATASET} --noise_type ${NOISE_TYPE} --seed ${SEED} > results/${DATASET}_${NOISE_TYPE}_seed_${SEED}/detection.log
````
and modify the arguments to your needs.

To train and evaluate our framework on *aggre*, *rand1* and *worse* label sets on the *detection* and *learning* task
to reproduce the results of the *Noisy Labels Challenge*, run 
```
bash run.sh
```

## Official Noisy Labels Leaderboard 
The [Noisy Label Challenge](http://noisylabels.com/) reports metrics based on runs with other settings in the training shedule. To run our framework with official challenge settings, you only need to set the validation split to 0, e.g. by modifying the above example to

````shell
export DATASET=cifar10
export NOISE_TYPE=aggre
export SEED=42
mkdir results
mkdir results/${DATASET}_${NOISE_TYPE}_seed_${SEED}
python blind_knowledge_dist_training.py --val_ratio 0 --dataset ${DATASET} --noise_type ${NOISE_TYPE} --seed ${SEED} > results/${DATASET}_${NOISE_TYPE}_seed_${SEED}/training.log
python learning.py --dataset ${DATASET} --noise_type ${NOISE_TYPE} --seed ${SEED} > results/${DATASET}_${NOISE_TYPE}_seed_${SEED}/learning.log
````

You can find models and learning results for all challenges [here](https://www.tnt.uni-hannover.de/de/project/MPT/data/NoisyLabelChallenge/NoisyLabelChallenge.zip).

## Citation

---
If you use our work in your research, please cite our publication:

```text
@misc{https://doi.org/10.48550/arxiv.2211.11355,
  doi = {10.48550/ARXIV.2211.11355},
  url = {https://arxiv.org/abs/2211.11355},
  author = {Kaiser, Timo and Ehmann, Lukas and Reinders, Christoph and Rosenhahn, Bodo}
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Blind Knowledge Distillation for Robust Image Classification},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
