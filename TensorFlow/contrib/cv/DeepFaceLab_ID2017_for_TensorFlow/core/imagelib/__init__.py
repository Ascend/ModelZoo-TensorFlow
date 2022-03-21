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

from .estimate_sharpness import estimate_sharpness

from .equalize_and_stack_square import equalize_and_stack_square

from .text import get_text_image, get_draw_text_lines

from .draw import draw_polygon, draw_rect

from .morph import morph_by_points

from .warp import gen_warp_params, warp_by_params

from .reduce_colors import reduce_colors

from .color_transfer import color_transfer, color_transfer_mix, color_transfer_sot, color_transfer_mkl, color_transfer_idt, color_hist_match, reinhard_color_transfer, linear_color_transfer

from .common import random_crop, normalize_channels, cut_odd_image, overlay_alpha_image

from .SegIEPolys import *

from .blursharpen import LinearMotionBlur, blursharpen

from .filters import apply_random_rgb_levels, \
                     apply_random_overlay_triangle, \
                     apply_random_hsv_shift, \
                     apply_random_sharpen, \
                     apply_random_motion_blur, \
                     apply_random_gaussian_blur, \
                     apply_random_nearest_resize, \
                     apply_random_bilinear_resize, \
                     apply_random_jpeg_compress, \
                     apply_random_relight
