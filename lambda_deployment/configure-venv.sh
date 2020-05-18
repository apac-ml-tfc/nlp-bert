#!/bin/bash

# Inspired by this blog post:
# https://segments.ai/blog/pytorch-on-lambda
#
# Adapted to split dependencies between /tmp and the Lambda packgae, since we have more than 512MB of libs

# Error out if any step fails:
set -e

ENVNAME=lambda_bert

if conda env list | grep -q $ENVNAME; then
    # TODO: A clear cache option with conda env remove -n $ENVNAME -y
    echo "Using existing conda env $ENVNAME"
else
    # Create and activate an empty conda env:
    conda create -n $ENVNAME python=3.6 -y
fi

source activate $ENVNAME

# Install dependencies:
# (Conda installs didn't seem to play ball e.g. conda install pytorch==1.4.0 cpuonly -c pytorch)
pip install torch==1.4.0+cpu -f 'https://download.pytorch.org/whl/torch_stable.html'
pip install numpy transformers==2.8
# pip install transformers==2.8

# Export from conda env to primary store in the Lambda build folder:
rm -rf lambda/build
mkdir -p lambda/build/packages-local
cp -R /home/ec2-user/anaconda3/envs/${ENVNAME}/lib/python3.6/site-packages/* lambda/build/packages-local/

cd lambda/build/packages-local

# Prune non-essentials:
find . -type d -name "tests" -exec rm -rf {} +
find . -type d -name "__pycache__" -exec rm -rf {} +
# Temporarily disabled to try and fix import errors, but doesn't seem to help yet:
# rm -rf ./{caffe2,wheel,wheel-*,pkg_resources,boto*,aws*,pip,pip-*,pipenv,setuptools}
# rm -rf ./{*.egg-info,*.dist-info}
find . -name \*.pyc -delete

cd ..

# Move selected (bulky) packages to the secondary store (unpacked to /tmp on init):
mkdir -p packages-tmpdir
# Played around with a few combinations here...
#for pkgname in torch torch-1.4.0+cpu.dist-info numpy numpy.libs numpy-1.18.4.dist-info
#for pkgname in torch torch-1.4.0+cpu.dist-info
for pkg in torch torch-1.4.0+cpu.dist-info numpy numpy.libs numpy-1.18.4.dist-info tokenizers tokenizers-0.5.2.dist-info
do
    mv packages-local/$pkg packages-tmpdir/$pkg
done

# Compress the tmpdir packages to keep the Lambda deployment zip small:
echo "Zipping packages..."
zip -q -r9 packages-tmpdir.zip packages-tmpdir
rm -r packages-tmpdir
# Note this zip **contains the packages-tmpdir folder in the root** because of our CWD

cd ../..

# Copy Lambda code into (the root of) the build folder:
cp -R lambda/src/. lambda/build/

# Now our lambda/build folder should contain:
# - The contents of lambda/src
# - a compressed packages-tmpdir.zip containing a packages-tmpdir folder of bulky dependencies
# - a packages-local folder of embedded dependencies

echo "Done!"
