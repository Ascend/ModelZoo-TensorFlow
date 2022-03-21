from sklearn.metrics import  average_precision_score, roc_auc_score
from sklearn.utils.multiclass import type_of_target
import numpy as np
import math
from sklearn.metrics import mean_squared_error


def aucPerformance(y_pred_afm, y_true):
    num_example=len(y_pred_afm)
    predictions_bounded = np.maximum(y_pred_afm, np.ones(num_example) * min(y_true))  # bound the lower values
    predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # bound the higher values
    RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
    print("Test RMSE: %.4f"%(RMSE))
    
def eval_om(label_bin_file, om_output):
    y_true = np.fromfile(label_bin_file,np.float64)
    predictions = np.fromfile(om_output,np.float32)
    num_example=len(predictions)
    y_pred_afm = np.reshape(predictions, (num_example,))
    aucPerformance(y_pred_afm,y_true)


eval_om("../data/y_true.bin","../data/afm_output_0.bin")