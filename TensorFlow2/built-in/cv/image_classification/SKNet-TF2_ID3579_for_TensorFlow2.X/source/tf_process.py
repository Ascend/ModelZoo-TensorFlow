import os, inspect
import time

import tensorflow as tf
import numpy as np

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def training(neuralnet, dataset, epochs, batch_size,log_steps,batch_nums,normalize=True,rank_size=1):

    # print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    iteration = 0

    test_sq = 20
    test_size = test_sq**2
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs))
        loss_list = []
        acc_list = []
        start_time = time.time()
        while(True):
            stime = time.time()   #计算TimeHistory的时间
            x_tr, y_tr, terminator = dataset.next_train(batch_size) # y_tr does not used in this prj.

            loss, accuracy, class_score = neuralnet.step(x=x_tr, y=y_tr, iteration=iteration, train=True)
            loss_list.append(loss)
            acc_list.append(accuracy)

            if iteration % log_steps == 0:
                # 添加时间日志
                now = time.time()
                elapsed_time = now - stime
                fps = batch_size * rank_size / (elapsed_time / log_steps)
                print("TimeHistory: %.3f seconds, %.3f examples/second, between steps %d and %d " % (
                elapsed_time, fps, iteration, iteration + log_steps))

            iteration += 1
            if(terminator): break

        # print("Epoch [%d / %d] (%d iteration)  Loss:%.5f, Acc:%.5f"%(epoch, epochs, iteration, loss, accuracy))
        end_time = time.time()
        epoch_time = end_time - start_time
        print('{}/{} - {:0.3f}s - loss: {:0.6f} - accuracy: {:0.6f}'.format(
                batch_nums, batch_nums, epoch_time, np.mean(loss_list), np.mean(acc_list)))

    neuralnet.save_params()

def test(neuralnet, dataset, batch_size):

    try: neuralnet.load_params()
    except: print("Parameter loading was failed")

    print("\nTest...")

    confusion_matrix = np.zeros((dataset.num_class, dataset.num_class), np.int32)
    while(True):
        x_te, y_te, terminator = dataset.next_test(1) # y_te does not used in this prj.
        loss, accuracy, class_score = neuralnet.step(x=x_te, y=y_te, train=False)

        label, logit = np.argmax(y_te[0]), np.argmax(class_score)
        confusion_matrix[label, logit] += 1

        if(terminator): break

    print("\nConfusion Matrix")
    print(confusion_matrix)

    tot_precision, tot_recall, tot_f1score = 0, 0, 0
    diagonal = 0
    for idx_c in range(dataset.num_class):
        precision = confusion_matrix[idx_c, idx_c] / np.sum(confusion_matrix[:, idx_c])
        recall = confusion_matrix[idx_c, idx_c] / np.sum(confusion_matrix[idx_c, :])
        f1socre = 2 * (precision * recall / (precision + recall))

        tot_precision += precision
        tot_recall += recall
        tot_f1score += f1socre
        diagonal += confusion_matrix[idx_c, idx_c]
        print("Class-%d | Precision: %.5f, Recall: %.5f, F1-Score: %.5f" \
            %(idx_c, precision, recall, f1socre))

    accuracy = diagonal / np.sum(confusion_matrix)
    print("\nTotal | Accuracy: %.5f, Precision: %.5f, Recall: %.5f, F1-Score: %.5f" \
        %(accuracy, tot_precision/dataset.num_class, tot_recall/dataset.num_class, tot_f1score/dataset.num_class))
