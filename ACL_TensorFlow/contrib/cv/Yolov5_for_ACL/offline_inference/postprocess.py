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


import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import OrderedDict
import time
import json
from tqdm import tqdm
import itertools
from tabulate import tabulate
from absl import app, flags
from absl.flags import FLAGS
import sys
sys.path.append('./')
from core.dataset import Dataset, DatasetFetcher
from core.utils import nms, read_class_names
from core.config import cfg

flags.DEFINE_string('model', 'yolov5', 'yolov3, yolov4, yolov5')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_integer('batchsize', 1, 'total batchsize across all gpus')
flags.DEFINE_string('data_path', './data/dataset/val2017.txt', 'path of annotation file')
flags.DEFINE_boolean('mosaic', True, 'activate mosaic data augmentation')


class COCOevaluator:
    def __init__(
            self,
            testset,
            input_size,
            num_class,
            cfg,
            inference_path):
        self.testset = testset
        self.inference_path = inference_path
        self.input_size = input_size
        self.num_class = num_class
        self.anno = cfg.TEST.ANNOT_PATH_
        self.dtfile = "./det_result.json"
        self.coco_dict = self.generate_det_json(self.testset, self.input_size, self.num_class, self.inference_path)

    def generate_det_json(self, testset, input_size, num_class, inference_path):

        supercategory = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, \
                         17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, \
                         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, \
                         51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, \
                         67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, \
                         86, 87, 88, 89, 90]  # actual label_id

        json_list = []
        fail_images = []
        pbar = tqdm(testset)
        iterdata = iter(pbar)
        fetcher = DatasetFetcher(testset)
        k = 1
        while True:
            try:
                annotations = next(iterdata)
            except StopIteration:
                break
            if annotations is None:
                break
            test_data, _, batch_image_id, scale, dw, dh = fetcher.process_annotations(annotations)
            pred_result = []
            for j in range(6):
                output_data = np.fromfile(inference_path + '/davinci_' + str(batch_image_id).zfill(12) + '_output' + str(j) + '.bin',
                                          dtype='float32')
                pred_result.append(output_data)
            k += 1

            for i in range(3):  # feature maps of 3 scales
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                conv = conv.reshape(FLAGS.batchsize, input_size // (8 * 2 ** i), input_size // (8 * 2 ** i), 255)
                conv = np.squeeze(conv).reshape(input_size // (8 * 2 ** i), input_size // (8 * 2 ** i), 3,
                                                5 + num_class)  # squeeze out 1st dimension due to batchsize 1
                pred = pred.reshape(FLAGS.batchsize, input_size // (8 * 2 ** i), input_size // (8 * 2 ** i), 3,
                             5 + num_class)
                scores_ = pred[..., 4:5].flatten()
                probs = np.max(pred[..., 5:], axis=-1).flatten()
                categories_ = np.argmax(conv[..., 5:], axis=-1).flatten()
                bboxes_ = pred[..., 0:4].reshape(-1, 4)
                scores_ = scores_ * probs  # conditional probablity

                scores = scores_ if i == 0 else np.concatenate((scores, scores_), axis=0)
                categories = categories_ if i == 0 else np.concatenate((categories, categories_), axis=0)
                bboxes = bboxes_ if i == 0 else np.concatenate((bboxes, bboxes_), axis=0)

            # max detections for one picture
            mask = (scores > 0.1)
            scores = scores[mask]
            categories = categories[mask]
            bboxes = bboxes[mask]

            # scale bboxes to origin shape
            bboxes[..., 0] = (bboxes[..., 0] - dw) / scale  # xcenter
            bboxes[..., 1] = (bboxes[..., 1] - dh) / scale  # ycenter
            bboxes[..., [2, 3]] = bboxes[..., [2, 3]] / scale  # w, h

            x1 = bboxes[..., 0] - bboxes[..., 2] / 2  # xmin
            y1 = bboxes[..., 1] - bboxes[..., 3] / 2  # ymin
            x2 = bboxes[..., 0] + bboxes[..., 2] / 2  # xmax
            y2 = bboxes[..., 1] + bboxes[..., 3] / 2  # ymax

            pre_nms_boxes = np.stack((x1, y1, x2, y2, scores, categories), axis=1)
            if pre_nms_boxes.shape[0] == 0:
                fail_images.append(batch_image_id)
                continue
            print(pre_nms_boxes.shape[0])
            nstart = time.time()
            best_bboxes = nms(pre_nms_boxes, iou_threshold=0.6)
            print('ntime', time.time() - nstart)
            x1, y1, x2, y2, scores, cls = np.stack(best_bboxes, axis=1)
            categories = cls.astype(np.int32)
            w, h = x2 - x1, y2 - y1

            for n in range(len(scores)):
                coco_dict = OrderedDict()
                coco_dict["score"] = float(scores[n])
                coco_dict["image_id"] = int(batch_image_id)
                coco_dict["category_id"] = supercategory[categories[n]]
                coco_dict["bbox"] = [float(x1[n]), float(y1[n]), float(w[n]), float(h[n])]
                json_list.append(coco_dict)

        print('writing into json file...')
        with open('./det_result.json', 'w', encoding='utf-8') as f:
            json.dump(json_list, f, ensure_ascii=False)
        return json_list

    def evaluate(self):
        cocoGT = COCO(self.anno)
        cocoDT = cocoGT.loadRes(self.dtfile)
        cocoEval = COCOeval(cocoGT, cocoDT, 'bbox')

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        precisions = cocoEval.eval['precision']
        with open('./data/classes/coco.names', 'r') as f:
            class_names = [line.strip() for line in f]
        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float('nan')
            results_per_category.append(('{}'.format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left"
        )
        print(table)


def main(_argv):
    testset = Dataset(FLAGS, is_training=False)
    NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))
    inference_path = 'offline_inference/output_bins/Yolov5'
    evaluator = COCOevaluator(testset, cfg.TRAIN.INPUT_SIZE, NUM_CLASS, cfg, inference_path)
    evaluator.evaluate()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
