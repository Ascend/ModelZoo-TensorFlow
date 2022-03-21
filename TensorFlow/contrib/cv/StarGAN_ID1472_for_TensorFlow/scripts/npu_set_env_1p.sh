export RANK_INDEX=0
export RANK_SIZE=1
export RANK_ID=0
export DEVICE_ID=7
export DEVICE_INDEX=0

export install_path=/usr/local/Ascend
#nnae
export LD_LIBRARY_PATH=${install_path}/nnae/latest/fwkacllib/lib64:$LD_LIBRARY_PATH
export PATH=${install_path}/nnae/latest/fwkacllib/ccec_compiler/bin:${install_path}/nnae/latest/fwkacllib/bin:$PATH
export PYTHONPATH=${install_path}/nnae/latest/fwkacllib/python/site-packages:$PYTHONPATH
export ASCEND_OPP_PATH=${install_path}/ascend-toolkit/latest/arm64-linux/opp
export ASCEND_AICPU_PATH=${install_path}/ascend-toolkit/latest
#tfplugin
export PYTHONPATH=${install_path}/tfplugin/latest/tfplugin/python/site-packages:$PYTHONPATH
#Ascend-dmi
export LD_LIBRARY_PATH=/usr/local/dcmi:${install_path}/toolbox/latest/Ascend-DMI/lib64:${LD_LIBRARY_PATH}
export PATH=${install_path}/toolbox/latest/Ascend-DMI/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/gcc7.3.0/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/gcc7.3.0/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib/python3.7/site-packages/torch/lib/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/usr/include/hdf5/lib
