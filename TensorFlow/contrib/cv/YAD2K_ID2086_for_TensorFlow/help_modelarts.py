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
import os
import datetime
import moxing as mox


def obs_data2modelarts(config):
    """
    Copy train data from obs to modelarts by using moxing api.
    """
    print("+++++++++++++++++begin running obs_data2modelarts method +++++++++++++")
    start = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(config.data_url, config.modelarts_data_dir))
    # # OBS copy to ModelArts   the former is obs's data  push to  the latter ModelArts's filename(we can not see)
    mox.file.copy_parallel(src_url=config.data_url, dst_url=config.modelarts_data_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from obs to modelarts, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(config.modelarts_data_dir)
    print("++++++++++++ ModelArts what we have +++++++++++++")
    print("===>>>Files:", files)


def modelarts_result2obs(FLAGS):
    """
    Copy debug data from modelarts to obs.
    According to the swich flags, the debug data may contains auto tune repository, 
    dump data for precision comparision, even the computation graph and profiling data.
    """
    work_dir = os.getcwd()
    print("+++++++ work_dir++++++", work_dir)
    ## copy result from modelarts to obs
    obs_result_dir = os.path.join(FLAGS.obs_dir, 'result')
    print("++++ obs_result_dir +++ ", obs_result_dir)
    if not mox.file.exists(obs_result_dir):
        mox.file.make_dirs(obs_result_dir)
    # # After training   ModelArts copy to OBS
    mox.file.copy_parallel(src_url=FLAGS.result, dst_url=obs_result_dir)
    print("===>>>Copy Event or Checkpoint from modelarts dir:{} to obs:{}".format(FLAGS.result, obs_result_dir))

    files_result = os.listdir(FLAGS.result)
    print("++++++++++++ ModelArts file we have +++++++++++++")
    print("===>>>FLAGS.result", files_result)

    # files_result2 = os.listdir(obs_result_dir)
    print("++++++++++++ ModelArt flect to obs  what file we have  +++++++++++++")
    # print("===>>>obs_result_dir", files_result2)

    ## Copy profiling data. Comment this snippets if npu_profiling is off.
    if FLAGS.profiling:
        modelarts_profiling_dir = os.path.join(work_dir, "npu_profiling")
        obs_profiling_dir = os.path.join(FLAGS.obs_dir, 'npu_profiling')
        if not mox.file.exists(obs_profiling_dir):
            mox.file.make_dirs(obs_profiling_dir)
        print(" modelarts_profiling_dir  file what we have ++ ")
        print("===>>>modelarts_profiling_dir", modelarts_profiling_dir)
        # # After training   ModelArts copy to OBS
        mox.file.copy_parallel(modelarts_profiling_dir, obs_profiling_dir)
        print("===>>>Profiling data:{} on OBS dir:{}".format(mox.file.list_directory(obs_profiling_dir),
                                                             obs_profiling_dir))
