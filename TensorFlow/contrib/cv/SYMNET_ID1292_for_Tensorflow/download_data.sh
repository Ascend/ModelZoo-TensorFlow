# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
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

mkdir tmp

# Download everything
mwget --show-progress -O tmp/attr-ops-data.tar.gz https://www.cs.utexas.edu/~tushar/attribute-ops/attr-ops-data.tar.gz
mwget --show-progress -O tmp/mitstates.zip http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip
mwget --show-progress -O tmp/utzap.zip http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip
mwget --show-progress -O tmp/natural.tar.gz http://www.cs.cmu.edu/~spurushw/publication/compositional/compositional_split_natural.tar.gz
echo "Data downloaded. Extracting files..."


# Dataset metadata and features
tar -zxvf tmp/attr-ops-data.tar.gz --strip 1
mv data/mit-states mit-states-original
mv data/ut-zap50k ut-zap50k-original
rm -r cv tensor-completion data



# MIT-States
unzip tmp/mitstates.zip 'release_dataset/images/*' -d mit-states-original/
mv mit-states-original/release_dataset/images mit-states-original/images/
rm -r mit-states-original/release_dataset
rename "s/ /_/g" mit-states-original/images/*

# UT-Zappos50k
unzip tmp/utzap.zip -d ut-zap50k-original/
mv ut-zap50k-original/ut-zap50k-images ut-zap50k-original/_images/
python reorganize_utzap.py
rm -r ut-zap50k-original/_images


# Natural split
tar -zxvf tmp/natural.tar.gz
mv mit-states/metadata_compositional-split-natural.t7 mit-states/metadata.t7
mv ut-zap50k/metadata_compositional-split-natural.t7 ut-zap50k/metadata.t7
mv mit-states/compositional-split-natural mit-states/compositional-split
mv ut-zap50k/compositional-split-natural ut-zap50k/compositional-split
mv mit-states mit-states-natural
mv ut-zap50k ut-zap50k-natural
ln -s ../mit-states-original/images mit-states-natural/images
ln -s ../ut-zap50k-original/images ut-zap50k-natural/images


# remove all zip files and temporary files
rm -r tmp
