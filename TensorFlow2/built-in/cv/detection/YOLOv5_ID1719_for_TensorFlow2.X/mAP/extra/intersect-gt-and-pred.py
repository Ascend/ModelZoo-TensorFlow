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

import sys
import os
import glob

## This script ensures same number of files in ground-truth and predicted folder.
## When you encounter file not found error, it's usually because you have
## mismatched numbers of ground-truth and predicted files.
## You can use this script to move ground-truth and predicted files that are
## not in the intersection into a backup folder (backup_no_matches_found).
## This will retain only files that have the same name in both folders.

# change directory to the one with the files to be changed
path_to_gt = '../ground-truth'
path_to_pred = '../predicted'
backup_folder = 'backup_no_matches_found'  # must end without slash

os.chdir(path_to_gt)
gt_files = glob.glob('*.txt')
if len(gt_files) == 0:
    print("Error: no .txt files found in", path_to_gt)
    sys.exit()
os.chdir(path_to_pred)
pred_files = glob.glob('*.txt')
if len(pred_files) == 0:
    print("Error: no .txt files found in", path_to_pred)
    sys.exit()

gt_files = set(gt_files)
pred_files = set(pred_files)
print('total ground-truth files:', len(gt_files))
print('total predicted files:', len(pred_files))
print()

gt_backup = gt_files - pred_files
pred_backup = pred_files - gt_files


def backup(src_folder, backup_files, backup_folder):
    # non-intersection files (txt format) will be moved to a backup folder
    if not backup_files:
        print('No backup required for', src_folder)
        return
    os.chdir(src_folder)
    ## create the backup dir if it doesn't exist already
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    for file in backup_files:
        os.rename(file, backup_folder + '/' + file)


backup(path_to_gt, gt_backup, backup_folder)
backup(path_to_pred, pred_backup, backup_folder)
if gt_backup:
    print('total ground-truth backup files:', len(gt_backup))
if pred_backup:
    print('total predicted backup files:', len(pred_backup))

intersection = gt_files & pred_files
print('total intersected files:', len(intersection))
print("Intersection completed!")