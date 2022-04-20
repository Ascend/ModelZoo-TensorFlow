import os
import sys
import copy
from collections import OrderedDict
import json


def parse_json(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f, object_pairs_hook=OrderedDict)
    return config


def generate_json(device_list, config, target_file):
    new_config = copy.deepcopy(config)
    device_insts = []
    insta_list = new_config["group_list"][0]["instance_list"]
    rank = 0
    for inst in insta_list:
        if inst["devices"][0]["device_id"] in device_list:
            inst["rank_id"] = str(rank)
            device_insts.append(inst)
            rank += 1
    new_config["group_list"][0]["device_num"] = str(rank)
    new_config["group_list"][0]["instance_count"] = str(rank)
    new_config["group_list"][0]["instance_list"] = device_insts

    print(f'[INFO] Writing out hccl config json file to {target_file}')
    with open(target_file, 'w') as f:
        json.dump(new_config, f)


if __name__ == '__main__':
    device_lists = sys.argv[1]
    source_json_file = sys.argv[2]
    target_file = sys.argv[3]

    device_lists = device_lists.strip().split(',')
    config = parse_json(source_json_file)

    generate_json(device_lists, config, target_file)
