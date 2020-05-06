#!/bin/bash

# Inspired by this blog post:
# https://segments.ai/blog/pytorch-on-lambda

# TODO: If exists delete:
# conda env remove -n lambda_bert -y

# Error out if any step fails:
set -e

ENVNAME=lambda_bert

# Create and activate an empty conda env:
conda create -n $ENVNAME python=3.6 -y
source activate $ENVNAME

# Install PyTorch in it (CPU only to reduce size):
#conda install pytorch==1.4.0 cpuonly -c pytorch

# Not this one as suggested by the original:
pip install torch==1.4.0+cpu -f 'https://download.pytorch.org/whl/torch_stable.html'
#pip install transformers==2.8

mkdir -p lambda/packages
cp -R /home/ec2-user/anaconda3/envs/${ENVNAME}/lib/python3.6/site-packages/* lambda/packages/

cd lambda/packages

# Prune non-essentials:
find . -type d -name "tests" -exec rm -rf {} +
find . -type d -name "__pycache__" -exec rm -rf {} +
rm -rf ./{caffe2,wheel,wheel-*,pkg_resources,boto*,aws*,pip,pip-*,pipenv,setuptools}
rm -rf ./{*.egg-info,*.dist-info}
find . -name \*.pyc -delete

# Zip up selected dependencies:
zip -r9 torch.zip torch
rm -r torch

# TODO: Install and zip transformers too? Explodes the dependencies...
