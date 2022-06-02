import argparse
import numpy as np
from model.rgbd import MTV
from utils import process_data
from metric import thrC, post_proC, err_rate
from metric import normalized_mutual_info_score, f1_score, rand_index_score, adjusted_rand_score

import tensorflow as tf
import os

import scipy.io as sio
import time

parser = argparse.ArgumentParser(description='Multi-view Deep Subspace CLustering Networks')
parser.add_argument('--path', metavar='DIR', default='./Data/rgbd_mtv.mat',
                    help='path to dataset')

parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--pretrain', default=10000, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--lr', default=1e-3, type=float,
                    help='number of total epochs to run')

parser.add_argument('--gpu', default='0', type=str,
                    help='GPU id to use.')

parser.add_argument('--ft', action='store_true', help='finetune')

parser.add_argument('--test', action='store_true', help='run kmeans on learned coef')

np.random.seed(1)
tf.compat.v1.set_random_seed(1)

                    
def main():
    args = parser.parse_args()
    np.random.seed(1)
    tf.compat.v1.set_random_seed(1)
    # ignore tensorflow warning
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第二块GPU（从0开始）
    
    view_shape, views, label = process_data(args)
    num_class = np.unique(label).shape[0] 
    batch_size = label.shape[0] 
    # class_single = batch_size / num_class # 10

    reg1 = 1.0
    reg2 = 1.0 
    alpha = max(0.4 - (num_class-1)/10 * 0.1, 0.1)
    lr = args.lr
    acc_= []
    
    tf.compat.v1.reset_default_graph()

    if args.test:
        label_10_subjs = label - label.min() + 1
        label_10_subjs = np.squeeze(label_10_subjs) 
        Coef = sio.loadmat('./result/rgbd_coef.mat')['coef']
        print('load mat ..')
        y_x, L = post_proC(Coef, label_10_subjs.max(), 3, 1)   
        missrate_x = err_rate(label_10_subjs, y_x)                
        acc_x = 1 - missrate_x
        nmi = normalized_mutual_info_score(label_10_subjs, y_x)
        f_measure = f1_score(label_10_subjs, y_x)
        ri = rand_index_score(label_10_subjs, y_x)
        ar = adjusted_rand_score(label_10_subjs, y_x)
        print("Final Accuracy accuracy %.4f " % acc_x)
        print("nmi: %.4f" % nmi, \
            "accuracy: %.4f" % acc_x, \
            "F-measure: %.4f" % f_measure, \
            "RI: %.4f" % ri, \
            "AR: %.4f" % ar)   
        exit()

    if not args.ft:
        # pretrian stage 
        mtv = MTV(view_shape=view_shape, batch_size=batch_size, ft=False, reg_constant1=reg1, reg_constant2=reg2)
        mtv.restore()        
        epoch = 0 
        min_loss = 9970
        while epoch < args.pretrain:
            loss = mtv.reconstruct(views[0], views[1], lr)
            print("epoch: %.1d" % epoch, "loss: %.8f" % (loss/float(batch_size)))
            if loss/float(batch_size) < min_loss:
                print('save model.')
                mtv.save_model()
                min_loss = loss/float(batch_size)                          
            epoch += 1
    else:
        t1=time.time()
        # self-expressive stage
        mtv = MTV(view_shape=view_shape, batch_size=batch_size, ft=True, reg_constant1=reg1, reg_constant2=reg2)
        mtv.restore()
        Coef = None
        label_10_subjs = label - label.min() + 1
        label_10_subjs = np.squeeze(label_10_subjs) 

        best_acc, best_epoch = 0, 0
        
        epoch = 0
        t = 0
        while epoch < args.epochs:
            t3=time.time()
            loss, Coef, Coef_1, Coef_2 = mtv.finetune(views[0], views[1], lr)
            print("epoch: %.1d" % epoch, "loss: %.8f" % (loss))
            epoch += 1
            t4=time.time()
            t = t+(t4-t3)
            print("The time of one epoch is: ", t4-t3)
        print("The pretrain time is:", t)

        Coef = thrC(Coef, alpha)                                  
        sio.savemat('./result/rgbd_coef.mat', dict([('coef', Coef)]))
        y_x, L = post_proC(Coef, label_10_subjs.max(), 3, 1)    
        missrate_x = err_rate(label_10_subjs, y_x)                
        acc_x = 1 - missrate_x
        nmi = normalized_mutual_info_score(label_10_subjs, y_x)
        f_measure = f1_score(label_10_subjs, y_x)
        ri = rand_index_score(label_10_subjs, y_x)
        ar = adjusted_rand_score(label_10_subjs, y_x)
        print("Final Accuracy accuracy %.4f " % acc_x)
        print("epoch: %d" % epoch, \
            "nmi: %.4f" % nmi, \
            "accuracy: %.4f" % acc_x, \
            "F-measure: %.4f" % f_measure, \
            "RI: %.4f" % ri, \
            "AR: %.4f" % ar)
        t2=time.time()
        print("The overal time is: ", t2-t1)

if __name__ == '__main__':
    main()