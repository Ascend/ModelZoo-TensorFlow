#!bin/bash
cur_path=`pwd`
echo "modify version file-aic-ascend910-ops-info.json"
#临时补丁，需要根据网络修改
cp $ASCEND_OPP_PATH/op_impl/built-in/ai_core/tbe/config/ascend910/aic-ascend910-ops-info.json $cur_path/aic-ascend910-ops-info.json.bak -f
python3 ops_info_patch.py