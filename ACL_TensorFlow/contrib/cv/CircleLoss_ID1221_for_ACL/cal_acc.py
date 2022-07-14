import os
import numpy as np
from tqdm import tqdm
import math
from sklearn.model_selection import KFold

output_path="/home/zhang/output/2022713_17_55_11_543761"
pairs_path="./pairs.txt"

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric == 0:

        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:

        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0,
                  subtract_mean=False):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])

    # print("nrof_pairs: ",nrof_pairs)
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy

def l2_normalize(x):
    epsilon = 1e-10
    norm=0
    for xe in x:
        norm+=xe*xe
    norm=max(math.sqrt(norm),epsilon)
    for i in range(len(x)):
        x[i]/=norm
    return x

def getEmbedding(path):
    fin=open(path,"r")
    line=fin.readline().split()
    for i in range(len(line)):
        line[i]=float(line[i])
    # return line
    return l2_normalize(line)

def getName(cls,n):
    n="%04d" % int(n)
    return cls+"_"+n+"_output_0.txt"

def eul_dist(x1,x2):
    res=0
    for i in range(len(x1)):
        res+=(x1[i]-x2[i])*(x1[i]-x2[i])
    return math.sqrt(res)

fp=open(pairs_path,"r")
lines=fp.readlines()
lines=lines[1:]
embeddings1=[]
embeddings2=[]
actual_issame=[]

print("loading data......")
for line in tqdm(lines):
    line=line.strip().split()
    if len(line)==3:
        cls=line[0]
        name1=getName(cls,line[1])
        name2=getName(cls,line[2])
        embedding1=getEmbedding(os.path.join(output_path,name1))
        embedding2=getEmbedding(os.path.join(output_path,name2))
        embeddings1.append(embedding1)
        embeddings2.append(embedding2)
        actual_issame.append(1)
    elif len(line)==4:
        cls1=line[0]
        cls2=line[2]
        name1=getName(cls1,line[1])
        name2=getName(cls2,line[3])
        embedding1=getEmbedding(os.path.join(output_path,name1))
        embedding2=getEmbedding(os.path.join(output_path,name2))
        embeddings1.append(embedding1)
        embeddings2.append(embedding2)
        actual_issame.append(0)

embeddings1=np.array(embeddings1)
embeddings2=np.array(embeddings2)

print("calculate accuracy.....")
thresholds = np.arange(0, 4, 0.01)
tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
                                           np.asarray(actual_issame), nrof_folds=10,
                                           distance_metric=0, subtract_mean=False)

print("accuracy: ",accuracy.mean())




