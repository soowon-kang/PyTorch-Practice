# PyTorch-Practice
Beginner level materials with PyTorch


# Download dataset
 - '02_DNN.ipynb' >> House Price Prediction
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

 - '03_CNN.ipynb' >> CIFAR-10: Small Image Classification
https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz

=====

# Troubleshooting

## Check the current kernel list
>>> jupyter kernelspec list

## Check conda environment
>>> conda env list

## Set the conda environment
>>> conda create -n <MY_CONDA_ENV_NAME> python=3.12
>>> conda activate <MY_CONDA_ENV_NAME>

## Install conda kernel package
>>> conda install -c anaconda ipykernel

## Register the new kernel
>>> python -m ipykernel install --user --name <MY_KERNEL_NAME> --display-name "<DISPLAY_NAME>"

ref: https://velog.io/@baeyuna97/jupyter-notebook%EC%97%90%EC%84%9C-anaconda-env%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0

## CUDA 12.6
ref: https://developer.nvidia.com/cuda-downloads

## cuDNN
ref: https://developer.nvidia.com/
ref: https://developer.nvidia.com/rdp/cudnn-download

## Check windows environment variables

## Confirm the settings
>>> nvcc --version

ref: https://lonaru-burnout.tistory.com/16

## Install PyTorch
>>> conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

ref: https://pytorch.org/get-started/locally/#windows-anaconda

## Install helpful module while using PyTorch
>>> conda install conda-forge::torchinfo

ref: https://anaconda.org/conda-forge/torchinfo

=========================================================

## Memo
PyTorch intenally uses `multiprocessing` module where the Tensors set `num_worker > 1`.
Unfortunately, the Jupyter Notebook of Windows version do not support that multiprocessing.
Please use WSL (Linux) or the separate `*.py`.

ref: https://newsight.tistory.com/323
