#!/usr/bin/env bash

set -e

# install dependecies
sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-tk

# setup virtualenv
python3 -m venv venv

# activate virtual env (for deactive use `deactivate`)
. venv/bin/activate

# run in virtualenv
pip install gaft \
            numpy \
            matplotlib
