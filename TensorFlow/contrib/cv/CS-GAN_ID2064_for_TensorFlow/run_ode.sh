#!/bin/sh
# Copyright 2019 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Install python3.5
which python3.5
if  [ $? -eq 1 ]; then
  echo 'Installing python3.5'
  (cd /usr/src/
   sudo wget https://www.python.org/ftp/python/3.5.6/Python-3.5.6.tgz
   tar -xvzf Python-3.5.6.tgz
   sudo tar -xvzf Python-3.5.6.tgz
   cd Python-3.5.6
   ./configure --enable-loadable-sqlite-extensions --enable-optimizations
   sudo make altinstall)
fi
# Fail on any error.
set -e
python3.5 -m venv cs_gan_venv
echo 'Created venv'
source cs_gan_venv/bin/activate
echo 'Installing pip'
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.5 get-pip.py pip==20.2.3

echo 'Getting requirements.'
pip install -r cs_gan/requirements.txt


echo 'Starting training...'
python3.5 -m cs_gan.main_ode
