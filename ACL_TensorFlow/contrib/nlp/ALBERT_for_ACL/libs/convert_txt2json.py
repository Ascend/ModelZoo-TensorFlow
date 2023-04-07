#针对lcqmc数据集，做txt到json文件转换
import json

def convert_txt2Json(txt_path, json_path):
    '''
        txt_path为lcqmc数据集dev.txt路径
        json_path为输出json文件的路径
    '''
    dev_txt = open(txt_path)
    lines = dev_txt.read().split('\n')
    lines.pop()
    with open(json_path, 'w') as f:
        for line in lines:
            json_line = line.split('\t')
            dict_line = {"sentence1": json_line[0], "sentence2": json_line[1], "label": json_line[2]}
            arr = json.dumps(dict_line)
            f.write(arr + '\n')
    dev_txt.close()

