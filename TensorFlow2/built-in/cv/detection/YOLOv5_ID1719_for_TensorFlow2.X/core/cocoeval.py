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

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import OrderedDict
import tensorflow as tf
import json
import numpy as np
from tqdm import tqdm
from core.dataset import DatasetFetcher
import threading
from tabulate import tabulate
import itertools
from .utils import nms

class COCOevaluator:
    def __init__(self, model, testset, input_size, num_class, flags):
        self.model = model
        self.testset = testset
        self.testset.rewind()
        self.input_size = input_size
        self.num_class = num_class
        self.anno = flags.gt_anno_path
        self.dtfile = './det_result.json'
        self.coco_dict = self.generate_det_json(self.model, self.testset, self.input_size, self.num_class)
        
    def generate_det_json(self, model, testset, input_size, num_class):
        supercategory = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, \
        17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, \
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, \
        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, \
        76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        
        json_list = []
        fail_images = []
        iterdata = iter(testset)
        data_iter = threading.Lock()
        gene = threading.Lock()
        run = threading.Lock()
        @tf.function
        def gen(test_data):
            pred_result = model(test_data, training=False)
            return pred_result

        def worker():
            fetcher = DatasetFetcher(testset)
            while True:
                with data_iter:
                    try:
                        annotations = next(iterdata)
                    except StopIteration:
                        break
                test_data, _, batch_image_id, scale, dw, dh = fetcher.process_annotations(annotations)
                with gene:
                    pred_result = gen(test_data)
                with run:
                    print('processing image No.', batch_image_id, flush=True)
                    for i in range(3): # feature maps of 3 scales
                        conv , pred = pred_result[i * 2], pred_result[i * 2 + 1]
                        conv = tf.reshape(conv.numpy().squeeze(), (input_size // (8 * 2 ** i), input_size // (8 * 2 ** i), 3, 5 + num_class))
                        
                        scores_ = pred[..., 4:5].numpy().flatten()
                        probs = np.max(pred[..., 5:], axis=-1).flatten()
                        categories_ = np.argmax(conv[..., 5:], axis=-1).flatten()
                        bboxes_ = pred[..., 0:4].numpy().reshape(-1, 4)
                        scores_ = scores_ * probs
                        
                        scores = scores_ if i == 0 else np.concatenate((scores, scores_), axis=0)
                        categories = categories_ if i == 0 else np.concatenate((categories, categories_), axis=0)
                        bboxes = bboxes_ if i == 0 else np.concatenate((bboxes, bboxes_), axis=0)
                        
                    # max detections for one picture
                    mask = (scores > 0.1)
                    scores = scores[mask]
                    categories = categories[mask]
                    bboxes = bboxes[mask]
                    
                    # scale bboxes to origin shape
                    bboxes[..., 0] = (bboxes[..., 0] - dw) / scale # x center`
                    bboxes[..., 1] = (bboxes[..., 1] - dh) / scale # y center
                    bboxes[..., [2, 3]] = bboxes[..., [2, 3]] / scale # w, h
                    
                    x1 = bboxes[..., 0] - bboxes[..., 2] / 2 # xmin
                    y1 = bboxes[..., 1] - bboxes[..., 3] / 2 # ymin
                    x2 = bboxes[..., 0] + bboxes[..., 2] / 2 # xmax
                    y2 = bboxes[..., 1] + bboxes[..., 3] / 2 # ymax

                    pre_nms_boxes = np.stack((x1, y1, x2, y2, scores, categories), axis=-1)
                    print('num bboxes before nms:', pre_nms_boxes.shape[0], flush=True)
                    if pre_nms_boxes.shape[0] == 0:
                        fail_images.append(batch_image_id)
                        continue
                    best_bboxes = nms(pre_nms_boxes, iou_threshold=0.3)
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
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        print('fail images:',fail_images)
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

        precisions = cocoEval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        with open('../../../data/classes/coco.names', 'r') as f:
            class_names = [line.strip() for line in f]
        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0 : all area ranges
            # max dets index -1 : typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate result
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        print(table)