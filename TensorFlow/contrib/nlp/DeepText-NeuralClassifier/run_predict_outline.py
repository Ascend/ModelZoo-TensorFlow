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
# ==============================================================================

from predict import Predictor
from config import Config
import sklearn.metrics
import pandas as pd
import numpy as np
import sys
#import copy
import os
import time

def load_txt():
    ff = open(config.data.test_text_file, encoding='utf8')
    txt = ff.read().split('\n')
    txt = txt[0:len(txt) - 1]
    ff.close()
    return txt

def load_tag():
    f_tag=open(config.data.dict_dir+'/label.dict',encoding='utf8')
    tag_dict=f_tag.readlines()
    tag_list_name=[]
    tag_list_num=[]
    for tag in (tag_dict):
        tag_list_name.append(tag.split('\t')[0])
        tag_list_num.append(tag.split('\t')[1])
    return tag_list_name,tag_list_num

def get_threshold(dict_map, probs, labels):
    threshold = []
    for idx_label in range(len(probs[0, :])):
        rightlabel = 0
        for i in range(len(labels)):
            for item in labels[i].split(','):
                if item == dict_map[idx_label]:
                    rightlabel += 1

        idx_list = probs[:, idx_label].argsort(axis=0)
        idx_list = idx_list[::-1]
        fscore_list = []
        TP = 0
        for i in range(len(labels)):

            true_label = labels[idx_list[i]].split(',')
            for m in range(len(true_label)):
                if true_label[m] == dict_map[idx_label]:
                    TP += 1
                    break
            if rightlabel==0:
                fscore_list.append(0)
                continue
            recall = TP / rightlabel
            prcs = TP / (i + 1)
            if (recall + prcs) == 0:
                fscore_list.append(0)
            else:
                fscore_list.append(2 * recall * prcs / (recall + prcs))

        threshold.append(probs[idx_list[fscore_list.index(max(fscore_list))], idx_label])
    f=open('result/threhold.txt','w+',encoding='utf-8')
    f.write(str(threshold))
    f.close()
    print (threshold)
    return threshold




def result_save():

    test_classname=[]
    test_content=[]
    for line in txt:
        line_str=line.split('\t')
        test_classname.append(line_str[0])
    
    prob_list=[]
    for i in range(len(probs_all)):
        prob_list.append(probs_all[i][0])
      

    probs=np.array(prob_list)
    multi_label= config.eval.multi_label
    threshold=0    #             config.eval.threshold
    result_dir='result'  #config.eval.acc_report_file

    data_name='test'  #config.eval.test 

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    confusion_matrix_file = os.path.join(result_dir, 'confusion_matrix_{}.csv'.format(data_name))
    classification_report_file = \
            os.path.join(result_dir, 'classification_report_{}{}.csv'.format(data_name,'_thr_{:0.2f}'.format(threshold) if multi_label else ''))
    acc_report_file = os.path.join(result_dir, 'acc_report_{}.txt'.format(data_name))
    test_result_file = os.path.join(result_dir, 'test_result_{}.csv'.format(data_name))

    #标签词典
    dict_class_int2string={}
    for  i in range(len(tag_dict)):
        dict_class_int2string[i]=tag_dict[i]
    print (str(dict_class_int2string))
    # 获得最优的阈值
    if multi_label:
    	#threshold=get_threshold(dict_class_int2string, probs, test_classname)
        threshold=np.load("result/threhold.npy")
    if multi_label:
        unknown_labels = set([y for x in set(test_classname) for y in x.split(',') if not y in dict_class_int2string.values()])
    else:
        unknown_labels = set([y for y in set(test_classname) if not y in dict_class_int2string.values()])
    known_label_count = len(dict_class_int2string)
    for i,v in enumerate(unknown_labels):
        dict_class_int2string[known_label_count+i] = v
    dict_class_string2int = dict(zip(dict_class_int2string.values(),dict_class_int2string.keys()))


    data={}
    if not multi_label:
        y_pred = probs.argmax(axis=1)
        y_true = [dict_class_string2int[x] for x in test_classname]
        miss = (y_pred!=y_true).astype(int).sum()
        data["predict_label"]= [dict_class_int2string[x] for x in y_pred]
        data["predict_confidence"]=probs.max(1)
    else:
        ##多标签矩阵
        I=np.eye(len(dict_class_int2string))
        y_true_labels=[[dict_class_string2int[y] for y in x.split(',')] for x in test_classname]
        y_true = np.array([sum(I[x].astype(int)).tolist() for x in y_true_labels])

        #多标签预测结果.  y_pred与y_true的shape不一致，y_true有y_pred未覆盖的label
        probs_th=(probs-threshold)/threshold
        y_pred=np.zeros_like(probs_th)
        for i in range(len(probs_th)):
            prob_bool=(probs_th[i]>=0).astype(int)
            if sum(prob_bool)==0:
                y_pred[i]=(probs_th[i]==max(probs_th[i])).astype(int)
            else:
               y_pred[i]=prob_bool
       #  y_pred =  (probs>=threshold).astype(int)
        y_pred =  np.append(y_pred,np.zeros((y_true.shape[0],y_true.shape[1]-y_pred.shape[1])),axis=1)
        miss = (y_true != y_pred).all(axis=1).astype(int).sum()
    
       # top_k = 3
       # top_k_index=probs.argsort(axis=1)[:,::-1][:,:top_k]
       # r,c=top_k_index.shape
       # xv, yv = np.meshgrid(range(c),range(r))
       # prob_top_k = probs[yv,top_k_index]
        data["predict_label"] = [','.join([dict_class_int2string[i]  for i,p in zip(range(len(yy)),yy) if p==1]) for yy  in  y_pred  ]
        data["predict_confidence"] = [','.join(['{:.6f}'.format(i)  for i,p in zip(pp,yy) if p==1]) for yy,pp  in zip(y_pred, probs)  ]

    #输出测试结果
    count = len(probs)
    accuracy = float(count-miss)/count
    acc_summary = "{}{}, dataset \n\tevaluate result :  count|miss|accuracy  =  {}|{}|{:.2%}\n".format(\
                  time.asctime(), \
                  ', multilabel', \
                  count, miss, accuracy)

    print(acc_summary)
    with open(acc_report_file,'a') as f:
        f.write(acc_summary)

    #输出原表格带预测列
    res = pd.DataFrame(data)
    columns = ["predict_label","predict_confidence"]
    columns =  columns + [x  for x in list(res.columns) if not x in columns]
    res[columns].to_csv(test_result_file,sep='\t',index=False,float_format='%.4f')

    target_names = [dict_class_int2string[x] for x in range(len(dict_class_int2string))]
  
    label_list_all = list(set(dict_class_int2string.values()))
    #评估报告
    if not classification_report_file is None:
        report = sklearn.metrics.classification_report(y_true=y_true, y_pred=y_pred, labels=range(len(target_names)), target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report).T
        #support改为int型
        report_df['support']=report_df['support'].astype(int)
        #按support排序，不排序average部分
        report_df=report_df.iloc[:len(target_names)].sort_values(by=['support'],ascending=False).append(report_df.iloc[len(target_names):])
        report_df.to_csv(classification_report_file,sep='\t',index=True,float_format='%.4f')

    #混淆矩阵
    if not multi_label  and  not confusion_matrix_file is None:
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true=y_true,y_pred=y_pred,labels=[x for x in range(len(dict_class_int2string))])
        #print("confusion_matrix={}".format(confusion_matrix))
        cm = pd.DataFrame(confusion_matrix,index=target_names,columns=target_names)
        cm.to_csv(confusion_matrix_file,sep='\t',index=True)

    print("done!")


#
#    f=open('prob_test.txt','w+',encoding='utf-8')
#    classification_report_file = 'data/classification_report_file.csv'
#    confusion_matrix_file ='data/confusion_matrix_file.csv'
#    right_count=np.zeros(len(tag_dict))
#    predict_count=np.zeros(len(tag_dict))
#    confusion_mat=np.zeros((len(tag_dict),len(tag_dict)))
#    standard_dict={}
#    taglist=[]
#    truetag= [i.split('\t')[0] for i in  txt]
#    for i in range(len(probs)):
#        f.write(str(probs[i][0]))
#        txt_split = txt[i].split('\t')
#        strlist = txt_split[1:]
#        tag = txt_split[0]
#        pre_label=(probs[i][0]).tolist()
#        pre_label_sort=copy.deepcopy(pre_label)
#        pre_label_sort.sort()
#        
#        Max_num=1
#        for i in range(Max_num):
#            tag_index=pre_label.index(pre_label_sort[-i-1])
#            taglist.append(tag_dict[tag_index])
#    f.close()
#    ff=open('diff','w+',encoding='utf-8')
#    for i in range(len(taglist)):
#        if taglist[i]!=truetag[i]:
#            ff.write(str(i)+'\t'+taglist[i]+'\t'+truetag[i]+'\n')
#    ff.close()    
#    
#    
#    labels=list(set(list(truetag)+list(taglist)))
#    report = sklearn.metrics.classification_report(y_true=truetag,y_pred=taglist, labels=labels, target_names=labels, output_dict=True)
#    report_df = pd.DataFrame(report).T
#    report_df.to_csv(classification_report_file,sep='\t',index=True,float_format='%.4f')
#
#    confusion_matrix = sklearn.metrics.confusion_matrix(y_true=truetag,y_pred=taglist,labels=labels)
#    #print("confusion_matrix={}".format(confusion_matrix))
#    cm = pd.DataFrame(confusion_matrix,index=labels,columns=labels)
#    cm.to_csv(confusion_matrix_file,sep='\t',index=True)
#
     
if __name__ == '__main__':
    config = Config(config_file=sys.argv[1])
    predictor = Predictor(config)
    txt=load_txt()
    tag_dict,tag_list_num=load_tag()
    probs_all = (predictor.predict(txt))
    result_save()

