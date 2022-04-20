#!/bin/bash
new_env=$1
cur_dir=`pwd`
root_dir=${cur_dir}

if [ ! -d ${new_env} ];then
      mkdir ${root_dir}/${new_env}
      cd ${root_dir}/${new_env}
      ln -s ../src src
      ln -s ../configs configs
      ln -s ../scripts scripts
fi
