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


from npu_bridge.npu_init import *
from avod.builders.dataset_builder import DatasetBuilder


def main(dataset=None):
    if not dataset:
        dataset = DatasetBuilder.build_kitti_dataset(
            DatasetBuilder.KITTI_TRAIN)

    label_cluster_utils = dataset.kitti_utils.label_cluster_utils

    print("Generating clusters in {}/{}".format(
        label_cluster_utils.data_dir, dataset.data_split))
    clusters, std_devs = dataset.get_cluster_info()

    print("Clusters generated")
    print("classes: {}".format(dataset.classes))
    print("num_clusters: {}".format(dataset.num_clusters))
    print("all_clusters:\n {}".format(clusters))
    print("all_std_devs:\n {}".format(std_devs))


if __name__ == '__main__':
    main()

