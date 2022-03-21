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


def _cmc_core(D, G, P):
    m, n = D.shape
    order = np.argsort(D, axis=0)
    match = (G[order] == P)
    print(order,match)
    print(match.sum(axis=1))
    print('cum')
    print(match.sum(axis=1) * 1.0 / n)
    print((match.sum(axis=1) * 1.0 / n).cumsum())
    return (match.sum(axis=1) * 1.0 / n).cumsum()


def cmc(distmat, glabels=None, plabels=None, ds=None, repeat=None):
    """Compute the Cumulative Match Characteristic (CMC)
    This function assumes that gallery labels have no duplication. If there are
    duplications, random downsampling will be performed on gallery labels, and
    the computation will be repeated to get an average result.
    Parameters
    ----------
    distmat : numpy.ndarray
        The distance matrix. ``distmat[i, j]`` is the distance between i-th
        gallery sample and j-th probe sample.
    glabels : numpy.ndarray or None, optional
    plabels : numpy.ndarray or None, optional
        If None, then gallery and probe labels are assumed to have no
        duplications. Otherwise, they represent the vector of gallery and probe
        labels. Default is None.
    ds : int or None, optional
        If None, then no downsampling on gallery labels will be performed.
        Otherwise, it represents the number of gallery labels to be randomly
        selected. Default is None.
    repeat : int or None, optional
        If None, then the function will repeat the computation for 100 times
        when downsampling is performed. Otherwise, it specifies the number of
        repetition. Default is None.
    Returns
    -------
    out : numpy.ndarray
        The rank-1 to rank-m accuracy, where m is the number of (downsampled)
        gallery labels.
    """
    m, n = distmat.shape
    print(m,n)
    if glabels is None and plabels is None:
        print ('1')
        glabels = np.arange(0, m)
        plabels = np.arange(0, n)
        print( glabels,plabels )
    if isinstance(glabels, list):
        print('2')
        glabels = np.asarray(glabels)
    if isinstance(plabels, list):
        print('3')
        plabels = np.asarray(plabels)
    ug = np.unique(glabels)
    print('ug:')
    print(ug)
    if ds is None:
        print('4')
        ds = ug.size
        print(ds)
    if repeat is None:
        print('5')
        if ds == ug.size and ug.size == len(glabels):
            print('55')
            repeat = 1
        else:
            repeat = 100

    ret = 0
    for __ in range(repeat):
        # Randomly select gallery labels
        G = np.random.choice(ug, ds, replace=False)
        # Select corresponding probe samples
        p_inds = [i for i in range(len(plabels)) if plabels[i] in G]
        P = plabels[p_inds]
        # Randomly select one gallery sample per label selected
        D = np.zeros((ds, P.size))
        print (G,P,D,p_inds)
        for i, g in enumerate(G):
            samples = np.where(glabels == g)[0]
            j = np.random.choice(samples)
            D[i, :] = distmat[j, p_inds]
        print ('dgp')
        print(D , G , P)
        # Compute CMC
        ret += _cmc_core(D, G, P)
    return ret / repeat
     
        
'''
gids = [0,1,1]
pids = [0,0,1,1]
D =np.matrix("0.8 0.9 0.3 0.5;0.3 0.5 0.6 0.1;0.9 0.1 0.1 0.1")
print(D)
print ( cmc(D, gids, pids) ) 
'''
'''
#p = np.array([[0.0, 1.0, 1.0],[1.0 ,0.0 ,1.0],[1.0,1.0,0.0]])
p = np.array([[0.1, 0.9],[0.85,0.9]])
a=np.ones((10, 10), dtype='f')
b=np.zeros((100, 100), dtype='f')
'''


'''
a[0,0]=0.0
a[1,1]=0.0
#a[2,2]=0.0
a[3,3]=0.0
a[4,4]=0.0
a[5,5]=0.0
a[6,6]=0.0

a[1,7]=0.0
a[8,7]=0.0
a[2,3]=0.0
a[1,2]=0.0
a[8,2]=0.0
a[5,6]=0.0
'''
