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
rm -rf ./{caffe2,wheel,wheel-*,pkg_resources,boto*,aws*,pip,pip-*,pipenv,setuptools}
rm -rf ./{*.egg-info,*.dist-info}
find . -name \*.pyc -delete

# Create the folder where we'll store our overflow dependencies:
cd ..
mkdir -p packages-tmpdir

# ...But which packages to leave in packages-local, and which to zip/extract to /tmp?
#
# For some reason (possibly relating to difference in architecture between SM notebook and Lambda host?)
# packages with binary lib files seem to error on import if installed in the (readonly) packages-local
# folder... So we'll select which packages go in to the tmpdir archive by choosing all those containing
# binary .so files.
#
# This is made less simple because some packages don't live in a subfolder (e.g. sentencepiece.py and
# _sentencepiece.cpython-36m-x86_64-linux-gnu.so directly in the site-packages folder), and we'd like to
# bring along metadata like .dist-info folders for any packages where it's present.
#
# The below creates a bar-separated list of base package names by:
# - `find`ing all .so files and printing the path beginning with the package (e.g. numpy/.../foobar.so)
# - `sed`ing to drop anything after the first /, . or - (and drop a leading newline if present)
# - Keeping only unique package names (in alphabetical order) with `sort`
# - Replacing newline separators with |, and then deleting the final | (since we had a trailing newline)
BINPKGS=`find ./packages-local -name "*.so" -printf "%P\n" | sed -nr "s/_?([^\/\.]*).*/\1/p" | sort -u | tr '\n' '|'`
BINPKGS=${BINPKGS::-1}

echo "\nMoving the following binary-containing packages to tmpdir: $BINPKGS"

# Now we **assume the package names don't contain any regex special characters**, and use our bar-separated
# list as the basis for a regex to find matching folders/objects. we print only the top-level folders/objects
# by our choice of find command input and printf result formatting.
#
# ...Then `sort` unique and use `awk`/`while` to iterate over the results. It's easier to construct our mv
# expression if we work from the packages-local folder itself:
cd packages-local
find ./* -regextype posix-extended -iregex "\.\/_?(${BINPKGS})[\.\/\-].*" -printf "%H\n" | sort -u | awk -F '\0' '{print $0}' | while read file;
do
    # :2 trims off the leading ./
    mv $file ../packages-tmpdir/${file:2}
    echo "moved ${file:2}"
done
cd ..

# Compress the tmpdir packages to keep the Lambda deployment zip small:
echo "Zipping packages..."
zip -q -r9 packages-tmpdir.zip packages-tmpdir
rm -rf packages-tmpdir
# Note this zip **contains the packages-tmpdir folder in the root** because of our CWD

cd ../..

# Copy Lambda code into (the root of) the build folder:
cp -R lambda/src/. lambda/build/

# Now our lambda/build folder should contain:
# - The contents of lambda/src
# - a compressed packages-tmpdir.zip containing a packages-tmpdir folder of bulky/binary dependencies
# - a packages-local folder of embedded dependencies

echo "Done!"
