#!/bin/bash
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

# convert the avi video to images
#   Usage (sudo for the remove priviledge):
#       sudo ./convert_video_to_images.sh path/to/video fps
#   Example Usage:
#       sudo ./convert_video_to_images.sh ~/document/videofile/ 5
#   Example Output:
#       ~/document/videofile/walk/video1.avi
#       #=>
#       ~/document/videofile/walk/video1/00001.jpg
#       ~/document/videofile/walk/video1/00002.jpg
#       ~/document/videofile/walk/video1/00003.jpg
#       ~/document/videofile/walk/video1/00004.jpg
#       ~/document/videofile/walk/video1/00005.jpg
#       ...

for folder in $1/*
do
    for file in "$folder"/*.avi
    do
        if [[ ! -d "${file[@]%.avi}" ]]; then
            mkdir -p "${file[@]%.avi}"
        fi
        ffmpeg -i "$file" -vf fps=$2 "${file[@]%.avi}"/%05d.jpg
        rm "$file"
    done
done