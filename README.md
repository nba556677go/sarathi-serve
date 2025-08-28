# Sarathi-Serve

This is the fork from [microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve.git). Sarathi is the backbone for profiling [vidur](https://github.com/nba556677go/vidur)


### Setup CUDA

Sarathi-Serve has been tested with CUDA 12.1 on A100 and A40 GPUs.

### checkout vidur branch
```sh
# checkout vidur branch
git checkout vidur
```

### Create mamba environment
Setup mamba if you don't already have it,

```sh
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh # follow the instructions from there
source ~/.bashrc #enable conda
```
### Enable enrironment for vidur profiling
* We have included dependencies from vidur repo for easier environment setup
1. Edit pip requirements path in sarathi-vidur-env/environment-dev.ym
```
# edit the pip path to both repo
  - pip:
    - --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/
    - -e /home/ec2-user/sarathi-serve
    - -r /home/ec2-user/sarathi-serve/requirements.txt
    - -r /home/ec2-user/vidur/requirements.txt 

```
2. Run creation
```
mamba env create -f sarathi-vidur-env/environment-dev.yml  
mamba activate sarathi-vidur
```
### [Optional] If you want to enable only sarathi environments, here are the setup steps

Create a Python 3.10 environment,

```sh
mamba create -p ./env python=3.10  
```

#### Install Sarathi-Serve

```sh
pip install -e . --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/
```

## Reproducing Results

Refer to readmes in individual folders corresponding to each figure in `osdi-experiments`.

## Citation

If you use our work, please consider citing our paper:

```
@article{agrawal2024taming,
  title={Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve},
  author={Agrawal, Amey and Kedia, Nitin and Panwar, Ashish and Mohan, Jayashree and Kwatra, Nipun and Gulavani, Bhargav S and Tumanov, Alexey and Ramjee, Ramachandran},
  journal={Proceedings of 18th USENIX Symposium on Operating Systems Design and Implementation, 2024, Santa Clara},
  year={2024}
}
```

## Acknowledgment

This repository originally started as a fork of the [vLLM project](https://vllm-project.github.io/). Sarathi-Serve is a research prototype and does not have complete feature parity with open-source vLLM. We have only retained the most critical features and adopted the codebase for faster research iterations.
