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

# #!/usr/bin/python
# # -*- encoding: utf-8 -*-
# import sys
# import os.path as osp
# import logging
# import pickle
# from tqdm import tqdm
# import numpy as np
# from data import loadMarketForTest

# FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
# logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
# logger = logging.getLogger(__name__)

# # import os
# # os.environ['CUDA_VISBLE_DEVICES'] = '3'

# import tensorflow as tf

# sess=tf.Session()
# #先加载图和参数变量
# saver = tf.train.import_meta_graph('/home/liulizhao/projects/liuyixin/projects/logs/fc2_adjustlr/model.ckpt-20500.meta')
# saver.restore(sess, tf.train.latest_checkpoint('/home/liulizhao/projects/liuyixin/projects/logs/fc2_adjustlr/'))
# # 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
# graph = tf.get_default_graph()
# x = graph.get_tensor_by_name("inputs:0")
# y = graph.get_tensor_by_name("labels:0")
# features = graph.get_tensor_by_name("features:0")

# # 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
# graph = tf.get_default_graph()

# def embed():
#     q,t = loadMarketForTest(data_dir='/home/liulizhao/projects/liuyixin/Market-1501-v15.09.15')
#     query_embds, query_pids, query_camids = q.batch_sample(10,2,all=False)
#     gallery_embds, gallery_pids, gallery_camids = t.batch_sample(750,3,all=False)

#     f1 = {x:query_embds,y:query_pids}
#     query_embds = sess.run(features,f1)
#     f2 = {x:gallery_embds,y:gallery_pids}
#     gallery_embds = sess.run(features,f2)

#     embd_res = (query_embds, query_pids, query_camids,gallery_embds, gallery_pids, gallery_camids)
    
#     with open('/home/liulizhao/projects/liuyixin/projects/checkpoint/embds.pkl', 'wb') as fw:
#         pickle.dump(embd_res, fw)
#     # logger.info('embedding done, dump to: /home/liulizhao/projects/liuyixin/projects/checkpoint/embds.pkl')
#     return embd_res


# def evaluate(embd_res, cmc_max_rank = 1):
#     query_embds, query_pids, query_camids, gallery_embds, gallery_pids, gallery_camids = embd_res

#     ## compute distance matrix
#     logger.info('compute distance matrix')
#     dist_mtx = np.matmul(query_embds, gallery_embds.T)
#     dist_mtx = 1.0 / (dist_mtx + 1)
#     n_q, n_g = dist_mtx.shape

#     logger.info('start evaluating ...')
#     indices = np.argsort(dist_mtx, axis = 1)
#     matches = gallery_pids[indices] == query_pids[:, np.newaxis]
#     matches = matches.astype(np.int32)
#     all_aps = []
#     all_cmcs = []
#     for query_idx in tqdm(range(n_q)):
#         query_pid = query_pids[query_idx]
#         query_camid = query_camids[query_idx]

#         ## exclude duplicated gallery pictures
#         order = indices[query_idx]
#         pid_diff = gallery_pids[order] != query_pid
#         camid_diff = gallery_camids[order] != query_camid
#         useful = gallery_pids[order] != -1
#         keep = np.logical_or(pid_diff, camid_diff)
#         keep = np.logical_and(keep, useful)
#         match = matches[query_idx][keep]

#         if not np.any(match): continue

#         ## compute cmc
#         cmc = match.cumsum()
#         cmc[cmc > 1] = 1
#         all_cmcs.append(cmc[:cmc_max_rank])

#         ## compute map
#         num_real = match.sum()
#         match_cum = match.cumsum()
#         match_cum = [el / (1.0 + i) for i, el in enumerate(match_cum)]
#         match_cum = np.array(match_cum) * match
#         ap = match_cum.sum() / num_real
#         all_aps.append(ap)

#     assert len(all_aps) > 0, "NO QUERRY APPEARS IN THE GALLERY"
#     mAP = sum(all_aps) / len(all_aps)
#     all_cmcs = np.array(all_cmcs, dtype = np.float32)
#     cmc = np.mean(all_cmcs, axis = 0)

#     return cmc, mAP


# if __name__ == '__main__':
#     embd_res = embed()
#     with open('/home/liulizhao/projects/liuyixin/projects/checkpoint/embds.pkl', 'rb') as fr:
#         embd_res = pickle.load(fr)

#     cmc, mAP = evaluate(embd_res)
#     print('cmc is: {}, map is: {}'.format(cmc, mAP))


import scipy.io
import torch
import numpy as np
#import time
import os


import argparse
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--filename',default='pytorch_result.mat', type=str)
parser.add_argument('--logdir',default='pytorch_result.mat', type=str)
opt = parser.parse_args()


#######################################################################
# Evaluate
def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

######################################################################
result = scipy.io.loadmat(opt.filename)
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

# query_feature = query_feature.cuda()
# gallery_feature = gallery_feature.cuda()

print(query_feature.shape)
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
#print(query_label)
for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    #print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
result_path = opt.logdir+"result.txt"
result_str = 'Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label))
print(result_str)
with open(result_path,'a') as f:    #设置文件对象
    f.write(result_str)                 #将字符串写入文件中

# multiple-query
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
if multi:
    for i in range(len(query_label)):
        mquery_index1 = np.argwhere(mquery_label==query_label[i])
        mquery_index2 = np.argwhere(mquery_cam==query_cam[i])
        mquery_index =  np.intersect1d(mquery_index1, mquery_index2)
        mq = torch.mean(mquery_feature[mquery_index,:], dim=0)
        ap_tmp, CMC_tmp = evaluate(mq,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        #print(i, CMC_tmp[0])
    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
