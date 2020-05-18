#!/bin/bash

# Install AWS SAM on a SageMaker notebook instance

# As prescribed in the SAM docs at the time of writing:
# https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install-linux.html

# Error on failure of any step:
set -e

# Install HomeBrew:
yes | sh -c "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install.sh)"
test -d ~/.linuxbrew && eval $(~/.linuxbrew/bin/brew shellenv)
test -d /home/linuxbrew/.linuxbrew && eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)
test -r ~/.bash_profile && echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.bash_profile
echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.profile

export PATH=$PATH:/home/linuxbrew/.linuxbrew/bin

brew --version

# Install SAM:
brew tap aws/tap
brew install aws-sam-cli

sam --version
