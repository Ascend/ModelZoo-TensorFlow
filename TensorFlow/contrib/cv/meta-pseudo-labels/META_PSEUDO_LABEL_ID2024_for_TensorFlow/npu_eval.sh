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

python -u main.py \
  --task_mode="eval" \
  --master="/root/projects/meta-pseudo-labels-tf1-gpu/results/worker" \
  --dataset_name="cifar10_4000_mpl" \
  --dataset_dir="/root/datasets/cifar-10-batches-bin" \
  --output_dir="/root/projects/meta-pseudo-labels-tf1-gpu/results/ema_ckpts" \
  --model_type="wrn-28-2" \
  --optim_type="momentum" \
  --lr_decay_type="cosine" \
  --nouse_augment \
  --alsologtostderr \
  --running_local_dev \
  --load_ema_checkpoint \
  --image_size=32 \
  --num_classes=10 \
  --log_every=50 \
  --save_every=100 \
  --train_batch_size=64 \
  --eval_batch_size=64 \
  --uda_data=7 \
  --weight_decay=5e-4 \
  --num_train_steps=300000 \
  --augment_magnitude=16 \
  --batch_norm_batch_size=256 \
  --dense_dropout_rate=0.2 \
  --ema_decay=0.995 \
  --label_smoothing=0.15 \
  --mpl_teacher_lr=0.05 \
  --mpl_teacher_lr_warmup_steps=5000 \
  --mpl_student_lr=0.05 \
  --mpl_student_lr_wait_steps=1000 \
  --mpl_student_lr_warmup_steps=5000 \
  --uda_steps=5000 \
  --uda_temp=0.7 \
  --uda_threshold=0.6 \
  --uda_weight=8
