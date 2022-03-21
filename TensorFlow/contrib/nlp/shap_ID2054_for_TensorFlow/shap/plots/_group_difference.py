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
import numpy as np
import warnings
try:
    import matplotlib.pyplot as pl
    import matplotlib
except ImportError:
    warnings.warn("matplotlib could not be loaded!")
    pass
from . import colors


def group_difference(shap_values, group_mask, feature_names=None, xlabel=None, xmin=None, xmax=None, 
                     max_display=None, sort=True, show=True):
    """ This plots the difference in mean SHAP values between two groups.
    
    It is useful to decompose many group level metrics about the model output among the
    input features. Quantitative fairness metrics for machine learning models are
    a common example of such group level metrics.
    
    Parameters
    ----------
    shap_values : numpy.array
        Matrix of SHAP values (# samples x # features) or a vector of model outputs (# samples).
        
    group_mask : numpy.array
        A boolean mask where True represents the first group of samples and False the second.
        
    feature_names : list
        A list of feature names.
    """
    
    # compute confidence bounds for the group difference value
    vs = []
    gmean = group_mask.mean()
    for i in range(200):
        r = np.random.rand(shap_values.shape[0]) > gmean
        vs.append(shap_values[r].mean(0) - shap_values[~r].mean(0))
    vs = np.array(vs)
    xerr = np.vstack([np.percentile(vs, 95, axis=0), np.percentile(vs, 5, axis=0)])
    
    # See if we were passed a single model output vector and not a matrix of SHAP values
    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(1, -1).T
        if feature_names is None:
            feature_names = [""]
    
    # fill in any missing feature names
    if feature_names is None:
        feature_names = ["Feature %d" % i for i in range(shap_values.shape[1])]

    diff = shap_values[group_mask].mean(0) - shap_values[~group_mask].mean(0)
    
    if sort is True:
        inds = np.argsort(-np.abs(diff)).astype(np.int)
    else:
        inds = np.arange(len(diff))
    
    if max_display is not None:
        inds = inds[:max_display]
    # draw the figure
    figsize = [6.4, 0.2 + 0.9 * len(inds)]
    pl.figure(figsize=figsize)
    ticks = range(len(inds)-1, -1, -1)
    pl.axvline(0, color="#999999", linewidth=0.5)
    pl.barh(
        ticks, diff[inds], color=colors.blue_rgb,
        capsize=3, xerr=np.abs(xerr[:,inds])
    )
    
    for i in range(len(inds)):
        pl.axhline(y=i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
    
    ax = pl.gca()
    ax.set_yticklabels([feature_names[i] for i in inds])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('none')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelsize=11)
    if xlabel is None:
        xlabel = "Group SHAP value difference"
    ax.set_xlabel(xlabel, fontsize=13)
    pl.yticks(ticks, fontsize=13)
    xlim = list(pl.xlim())
    if xmin is not None:
        xlim[0] = xmin
    if xmax is not None:
        xlim[1] = xmax
    pl.xlim(*xlim)
    if show:
        pl.show()
