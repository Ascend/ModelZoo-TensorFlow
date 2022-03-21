from sklearn.metrics import  average_precision_score, roc_auc_score
import numpy as np

def aucPerformance(mse, labels):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;

def eval_om(label_bin_file, om_output):
    test_label = np.fromfile(label_bin_file,dtype=np.int64)
    score = np.fromfile(om_output,dtype=np.float32)
    aucPerformance(score,test_label)

eval_om("../data/test_label.bin","../data/dev-net_output_0.bin")