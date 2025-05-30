# The code and data for our paper "Towards Reward Fairness in RLHF: from a Resource Allocation Perspective"

This repository contains the code for training the fairness reward model and DPO as described in our paper. The repository is organized into two main directories: `Fair-RM` and `Fair-DPO`.

## Fair-RM

### Environment Setup

To set up the environment for `Fair-RM`, follow these steps:

```bash
conda create -n rm python=3.10.9
conda activate rm
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout d17fd7cd3b71c6a7bf7af34d8dc73135bb7ea8e9
# The test cuda version is 12.1, 12.2. You may need to update the torch version based on your cuda version...
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .
pip install flash-attn==2.6.3
pip install accelerate==0.33.0 
pip install deepspeed==0.12.2
pip install transformers==4.43.4
pip install numpy==1.26.4 # Note that the numpy version should be `numpy<2.0`. `Numpy 2.0` will encounter unexpected issues!!!
```

### Running the Code

To run the `Fair-RM` code, navigate to the `Fair-RM` directory and execute the following script:

```bash
cd Fair-RM
bash run_llama3.sh
```

You can modify the parameters in `run_llama3.sh` as needed. 

- The `mode` parameter has three options: `bt`, `fr`, and `fc`, which stand for Bradley-Terry (BT) model, Fairness Regularization, and Fairness Coefficient, respectively. 

- The parameters `alpha`, `tau`, and `gamma` correspond to the three hyperparameters discussed in the paper.

## Fair-DPO

### Environment Setup

To set up the environment for `Fair-DPO`, follow these steps:

```bash
conda create --name dpo python=3.10.14
conda activate dpo
conda install pip
pip install packaging ninja
conda install pytorch=2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install flash-attn==2.6.3 --no-build-isolation
pip install transformers==4.46.2
pip install peft==0.12.0
pip install datasets==2.20.0
pip install accelerate==0.33.0
pip install vllm==0.6.3.post1
```

### Running the Code

To run the `Fair-DPO` code, navigate to the `Fair-DPO` directory and execute the following script:

```bash
cd Fair-DPO
bash dpo_launch.sh
```

You can modify the parameters in `dpo_launch.sh` as needed. 

- The `mode` parameter has three options: `dpo`, `fr`, and `fc`, which stand for native DPO, Fairness Regularization DPO, and Fairness Coefficient DPO, respectively. 

- The parameters `alpha`, `tau`, and `gamma` correspond to the three hyperparameters discussed in the paper.

## Acknowledgements

We would like to thank the following GitHub repositories for their contributions, which were instrumental in building our data and implementing our methods:

1. [ContextualAI/HALOs](https://github.com/ContextualAI/HALOs)
2. [RLHFlow/RLHF-Reward-Modeling](https://github.com/RLHFlow/RLHF-Reward-Modeling)
3. [anthropics/hh-rlhf](https://github.com/anthropics/hh-rlhf)
   
## Citation

```
@misc{ouyang2025rewardfairnessrlhfresource,
      title={Towards Reward Fairness in RLHF: From a Resource Allocation Perspective}, 
      author={Sheng Ouyang and Yulan Hu and Ge Chen and Qingyang Li and Fuzheng Zhang and Yong Liu},
      year={2025},
      eprint={2505.23349},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.23349}, 
}
```
