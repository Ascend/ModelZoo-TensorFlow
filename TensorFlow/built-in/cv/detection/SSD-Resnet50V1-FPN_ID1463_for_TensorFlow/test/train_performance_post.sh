#!bin/bash
cur_path=`pwd`
echo "Recover version file-aic-ascend910-ops-info.json"
#临时补丁回退，需要根据网络修改
cp $cur_path/aic-ascend910-ops-info.json.bak ${ASCEND_OPP_PATH}/op_impl/built-in/ai_core/tbe/config/ascend910/aic-ascend910-ops-info.json -f
