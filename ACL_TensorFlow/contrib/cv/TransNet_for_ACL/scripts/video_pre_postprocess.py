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

import ffmpeg
import numpy as np
import sys
import argparse
import os
from os.path import join

class TransNetParams:
    F = 16
    L = 3
    S = 2
    D = 256
    INPUT_WIDTH = 48
    INPUT_HEIGHT = 27

params = TransNetParams()

def predict_video(frames,output_dir):
    assert len(frames.shape) == 4 and \
           list(frames.shape[1:]) == [params.INPUT_HEIGHT, params.INPUT_WIDTH, 3], \
        "[TransNet] Input shape must be [frames, height, width, 3]."

    def input_iterator():
        # return windows of size 100 where the first/last 25 frames are from the previous/next batch
        # the first and last window must be padded by copies of the first and last frame of the video
        no_padded_frames_start = 25
        no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

        start_frame = np.expand_dims(frames[0], 0)
        end_frame = np.expand_dims(frames[-1], 0)
        padded_inputs = np.concatenate(
            [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
        )

        ptr = 0
        while ptr + 100 <= len(padded_inputs):
            out = padded_inputs[ptr:ptr + 100]
            ptr += 50
            yield out

    index = 0
    for inp in input_iterator():
        np.expand_dims(inp, 0).tofile(join(output_dir,"{}.bin".format(str(index).zfill(6))))
        print("\r[TransNet] Processing video frames {}/{}".format(min(index * 50, len(frames)), len(frames)))
        index += 1

def scenes_from_predictions(predictions, threshold = 0.1):
    predictions = (predictions > threshold).astype(np.uint8)

    scenes = []
    t, tp, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if tp == 1 and t == 0:
            start = i
        if tp == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        tp = t
    if t == 0:
        scenes.append([start, i])
    return np.array(scenes, dtype=np.int32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="video preprocess and postprocess")
    parser.add_argument("--video_path", type=str, default="./BigBuckBunny.mp4",help="The path of the original video file")
    parser.add_argument("--output_path", type=str, default="./input_bins",help="The output path of bin files")
    parser.add_argument("--predict_path", type=str, default='./output',help='The path of npu predict files')
    parser.add_argument("--mode", type=str, default='preprocess',choices=['preprocess','postprocess'] help='Whether execute preprocess or postprocess')
    args = parser.parse_args()

    # export video into numpy array using ffmpeg
    video_stream, err = (
        ffmpeg
            .input(args.video_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24',s='{}x{}'.format(params.INPUT_WIDTH, params.INPUT_HEIGHT))
            .run(capture_stdout=True)
    )
    video = np.frombuffer(video_stream, np.uint8).reshape([-1, params.INPUT_HEIGHT, params.INPUT_WIDTH, 3])

    if args.mode == 'preprocess':
        if not os.path.isdir(args.output_path):
            os.makedirs(args.output_path)
        predict_video(video,args.output_path)
    else:
        res = []
        files = os.listdir(args.predict_path)
        files.sort()
        for file in files:
            print("start to process {}".format(file))
            pred = np.fromfile(join(args.predict_path, file), dtype='float32').reshape(1, 100)[0, 25:75]
            res.append(pred)

        predictions = np.concatenate(res)[:len(video)]
        scenes = scenes_from_predictions(predictions, threshold = 0.1)
        print('*'*20,'Predict Result','*'*20)
        print(scenes) 
