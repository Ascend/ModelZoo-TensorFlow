export install_path=/usr/local/Ascend/nnae/latest
#driver包依赖
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH #仅容器训练场景配置
export LD_LIBRARY_PATH=/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
#fwkacllib 包依赖
export LD_LIBRARY_PATH=${install_path}/fwkacllib/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/fwkacllib/python/site-packages:${install_path}/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:${install_path}/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
export PATH=${install_path}/fwkacllib/ccec_compiler/bin:${install_path}/fwkacllib/bin:$PATH
#tfplugin 包依赖
export PYTHONPATH=/usr/local/Ascend/tfplugin/latest/tfplugin/python/site-packages:$PYTHONPATH
#opp包依赖
export ASCEND_OPP_PATH=${install_path}/opp

export 