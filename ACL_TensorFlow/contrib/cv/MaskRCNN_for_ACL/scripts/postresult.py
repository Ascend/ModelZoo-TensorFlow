# Copyright 2020 Huawei Technologies Co., Ltd
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

from absl import logging
import tensorflow.compat.v1 as tf
import sys,os
import six
import numpy as np

import coco_metric
import coco_utils
import om_to_predict

def process_prediction_for_eval(prediction):
  """Process the model prediction for COCO eval."""
  image_info = prediction['image_info']
  box_coordinates = prediction['detection_boxes']
  processed_box_coordinates = np.zeros_like(box_coordinates)

  for image_id in range(box_coordinates.shape[0]):
    scale = image_info[image_id][2]
    for box_id in range(box_coordinates.shape[1]):
      # Map [y1, x1, y2, x2] -> [x1, y1, w, h] and multiply detections
      # by image scale.
      y1, x1, y2, x2 = box_coordinates[image_id, box_id, :]
      new_box = scale * np.array([x1, y1, x2 - x1, y2 - y1])
      processed_box_coordinates[image_id, box_id, :] = new_box
  prediction['detection_boxes'] = processed_box_coordinates
  return prediction

def compute_coco_eval_metric(predictor,
                             include_mask=True,
                             annotation_json_file=None):


  if not annotation_json_file:
    annotation_json_file = None
  use_groundtruth_from_json = (annotation_json_file is not None)
  batch_idx = 0
  predictions = dict()
  while True:
    try:
      prediction = six.next(predictor)
      print('Running inference on batch %d...', (batch_idx + 1))
      logging.info('Running inference on batch %d...', (batch_idx + 1))
    except StopIteration:
      print('Finished the eval set at %d batch.', (batch_idx + 1))
      logging.info('Finished the eval set at %d batch.', (batch_idx + 1))
      break

    prediction = process_prediction_for_eval(prediction)
    for k, v in six.iteritems(prediction):
      if k not in predictions:
        predictions[k] = [v]
      else:
        predictions[k].append(v)

    batch_idx = batch_idx + 1

  for k, v in six.iteritems(predictions):
    predictions[k] = np.concatenate(predictions[k], axis=0)

  if 'orig_images' in predictions and predictions['orig_images'].shape[0] > 10:
    # Only samples a few images for visualization.
    predictions['orig_images'] = predictions['orig_images'][:10]

  if use_groundtruth_from_json:

    eval_metric = coco_metric.EvaluationMetric(
        annotation_json_file, include_mask=include_mask)
    eval_results = eval_metric.predict_metric_fn(predictions)

  else:
    images, annotations = coco_utils.extract_coco_groundtruth(
        predictions, include_mask)
    dataset = coco_utils.create_coco_format_dataset(images, annotations)
    eval_metric = coco_metric.EvaluationMetric(
        filename=None, include_mask=include_mask)
    eval_results = eval_metric.predict_metric_fn(
        predictions, groundtruth_data=dataset)
  print('Eval results: ', eval_results)
  logging.info('Eval results: %s', eval_results)
  return eval_results, predictions

if __name__ == '__main__':
    include_mask = True
    val_json_file = '{}/instances_val2017.json'.format(sys.path[0])
    modeoutput = om_to_predict.do_bm_predict(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
    predictor=iter(modeoutput)
    compute_coco_eval_metric(predictor, include_mask=include_mask, annotation_json_file = val_json_file)

