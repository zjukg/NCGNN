# Newton–Cotes Graph Neural Networks: On the Time Evolution of Dynamic Systems

## Introduction

Hi, this repository is the official implementation of [Newton–Cotes Graph Neural Networks: On the Time Evolution of Dynamic Systems](https://arxiv.org/abs/2305.14642), NeurIPS 2023.


## Quick Start


**1. Installation**


```bash
conda create -n ncgnn python=3.8 # create a new env
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia # install torch with cuda
conda install networkx, pandas # install other packages
conda matplotlib, seaborn # install other packages for EqMotion

```

**2. Run NC (GMN) on MD17 (Aspirin)**

```bash
python -u spatial_graph/main_ncgnn.py --config_by_file \
                --model gmn	 \
                --use_extra_data 0 \
                --n_step 2 \
                --dataset md17 \
                --l2_penalty 0 \
```


**3. Run NC+ (GMN)**

```bash
python -u spatial_graph/main_ncgnn.py --config_by_file \
                --model gmn	 \
                --use_extra_data 1 \
                --n_step 2 \
                --dataset md17 \
                --l2_penalty 1 \
```


**4. Run NC (EqMotion) on multi-step prediction**

```bash
cd EqMotion
python main_nbody_reasoning.py --ncgnn # run nbody reasoning task
python main_nbody.py --ncgnn # run nobdy prediction task
python main_md17.py --mol aspirin --ncgnn # run MD17 (Aspirin) prediction task 
```


## Run on Other Datasets

**1. N-body**

Generate the dataset identical to that used by GMN:

```bash
python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 3 --n_stick 2 --n_hinge 1 --n_workers 50
```

Run on the generated dataset:

```bash
python -u spatial_graph/main_ncgnn.py --config_by_file \
                --model gmn	 \
                --use_extra_data 1 \
                --n_step 2 \
                --dataset nbody \
                --n_isolated 3 --n_stick 2 --n_hinge 1 \
                --l2_penalty 1 \
```
Note that n_isolated, n_stick, and n_hinge should match the generation settings exactly.

**2. Full MD17 datasets**

The MD17 dataset can be downloaded from [MD17](http://quantum-machine.org/gdml/#datasets). Download the dataset and place the files in `spatial_graph/MD17` for single-step prediction and 'EqMotion/md17/dataset' for EqMotion.

**3. Motion Capture**

The dataset can be downloaded from [CMU Motion Capture Database](http://mocap.cs.cmu.edu/search.php?subjectnumber=35). 

Run NCGNN on the motion dataset:

```bash
python -u spatial_graph/main_ncgnn.py --config_by_file \
                --model gmn	 \
                --use_extra_data 1 \
                --n_step 2 \
                --dataset motion \
                --l2_penalty 1 \
```

## Special Thanks

This source code is adaped from the official repository of [GMN](https://github.com/hanjq17/GMN) and [EqMotion](https://github.com/MediaBrain-SJTU/EqMotion).


## Citation

```bib
@inproceedings{NCGNN,
author = {Lingbing Guo and
          Weiqing Wang and
          Zhuo Chen and
          Ningyu Zhang and
          Zequn Sun and
          Yixuan Lai and
          Qiang Zhang and
          Huajun Chen},
  title = {Newton-Cotes Graph Neural Networks: On the Time Evolution of Dynamic Systems},
  booktitle = {NeurIPS},
  year = {2023}
}
```