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
import sys

import numpy as np
from sklearn.cluster import KMeans

from wavedata.tools.obj_detection import obj_utils

from avod.builders.dataset_builder import DatasetBuilder
from avod.core.label_cluster_utils import LabelClusterUtils


def main():
    """
    Calculates clusters for each class

    Returns:
        all_clusters: list of clusters for each class
        all_std_devs: list of cluster standard deviations for each class
    """

    dataset = DatasetBuilder.build_kitti_dataset(DatasetBuilder.KITTI_TRAIN)

    # Calculate the remaining clusters
    # Load labels corresponding to the sample list for clustering
    sample_list = dataset.load_sample_names(dataset.cluster_split)
    all_dims = []

    num_samples = len(sample_list)
    for sample_idx in range(num_samples):

        sys.stdout.write("\rClustering labels {} / {}".format(
            sample_idx + 1, num_samples))
        sys.stdout.flush()

        sample_name = sample_list[sample_idx]
        img_idx = int(sample_name)

        obj_labels = obj_utils.read_labels(dataset.label_dir, img_idx)
        filtered_lwh = LabelClusterUtils._filter_labels_by_class(
                obj_labels, dataset.classes)

        if filtered_lwh[0]:
            all_dims.extend(filtered_lwh[0])

    all_dims = np.array(all_dims)
    print("\nFinished reading labels, clustering data...\n")

    # Print 3 decimal places
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Calculate average cluster
    k_means = KMeans(n_clusters=1,
                     random_state=0).fit(all_dims)

    cluster_centre = k_means.cluster_centers_[0]

    # Calculate std. dev
    std_dev = np.std(all_dims, axis=0)

    # Calculate 2 and 3 standard deviations below the mean
    two_sigma_length_lo = cluster_centre[0] - 2 * std_dev[0]
    three_sigma_length_lo = cluster_centre[0] - 3 * std_dev[0]

    # Remove all labels with length above two std dev
    # from the mean and re-cluster
    small_mask_2 = all_dims[:, 0] < two_sigma_length_lo
    small_dims_2 = all_dims[small_mask_2]

    small_mask_3 = all_dims[:, 0] < three_sigma_length_lo
    small_dims_3 = all_dims[small_mask_3]

    small_k_means_2 = KMeans(n_clusters=1, random_state=0).fit(small_dims_2)
    small_k_means_3 = KMeans(n_clusters=1, random_state=0).fit(small_dims_3)
    small_std_dev_2 = np.std(small_dims_2, axis=0)
    small_std_dev_3 = np.std(small_dims_3, axis=0)

    print('small_k_means_2:', small_k_means_2.cluster_centers_)
    print('small_k_means_3:', small_k_means_3.cluster_centers_)
    print('small_std_dev_2:', small_std_dev_2)
    print('small_std_dev_3:', small_std_dev_3)

    # Calculate 2 and 3 standard deviations above the mean
    two_sigma_length_hi = cluster_centre[0] + 2 * std_dev[0]
    three_sigma_length_hi = cluster_centre[0] + 3 * std_dev[0]

    # Remove all labels with length above two std dev
    # from the mean and re-cluster
    large_mask_2 = all_dims[:, 0] > two_sigma_length_hi
    large_dims_2 = all_dims[large_mask_2]

    large_mask_3 = all_dims[:, 0] > three_sigma_length_hi
    large_dims_3 = all_dims[large_mask_3]

    large_k_means_2 = KMeans(n_clusters=1, random_state=0).fit(large_dims_2)
    large_k_means_3 = KMeans(n_clusters=1, random_state=0).fit(large_dims_3)

    large_std_dev_2 = np.std(large_dims_2, axis=0)
    large_std_dev_3 = np.std(large_dims_3, axis=0)

    print('large_k_means_2:', large_k_means_2.cluster_centers_)
    print('large_k_means_3:', large_k_means_3.cluster_centers_)
    print('large_std_dev_2:', large_std_dev_2)
    print('large_std_dev_3:', large_std_dev_3)


if __name__ == '__main__':
    main()

