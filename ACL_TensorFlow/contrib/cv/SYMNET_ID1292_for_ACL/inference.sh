#!/bin/bash

Placeholder_2=./data/bin_file/Placeholder_2
test_att_id=./data/bin_file/test_att_id
test_obj_id=./data/bin_file/test_obj_id
Placeholder_6=./data/bin_file/Placeholder_6

om_path="/home/HwHiAiUser/AscendProjects/SYMNET_ID1292_for_ACL/data/om/symnet.om"
output_path=./data/output
ulimit -c 0
./msame --model ${om_path} --input ${Placeholder_2},${test_att_id},${test_obj_id},${Placeholder_6} --output ${output_path} --outfmt TXT
