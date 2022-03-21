/*******************************************************************************
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*******************************************************************************/

/*******************************************************************************
Created by wang-bain on 2021/3/18.
*******************************************************************************/


#include "acl/acl.h"

#include <algorithm>
#include <dirent.h>
#include <map>
#include <memory.h>
#include <string>
#include <vector>
#include <fstream>
#include <getopt.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;

#ifndef RUN_ACL_MODEL_UTILS_H
#define RUN_ACL_MODEL_UTILS_H

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

void Help();

string GetTime();

string FormatInt(string Idx, size_t formatSize);

Result CreateDir(string pathName);

void Split(const string &s, vector <string> &tokens, const string &delimiters);

int ScanFiles(vector <string> &fileList, string inputDirectory);

void MergeInputFile(vector <string> &fileList, string &outputFile);

void ReleaseAllModelResource(uint32_t modelId, uint32_t nodeId,
                             aclmdlDesc *modelDesc, vector <aclrtContext> &v_context);

aclmdlDesc *CreateDesc(aclmdlDesc *modelDesc, uint32_t modelId);

aclmdlDataset *CreateInput(aclmdlDesc *modelDesc, aclmdlDataset *input,
                           vector <string> v_inputFiles, uint32_t remoteDevice);

Result CreateInputDesc(aclmdlDataset *input, vector <string> v_inputFiles, string dynamicShape,
                       aclTensorDesc *inputDesc[]);

aclmdlDataset *CreateZeroInput(aclmdlDesc *modelDesc, aclmdlDataset *input,
                               vector <string> v_inputFiles, uint32_t remoteDevice);

aclmdlDataset *CreateOutput(aclmdlDesc *modelDesc, aclmdlDataset *output, aclmdlDataset *input, string dynamicOutSize);

Result InitResource(uint32_t nodeId, vector <aclrtContext> &v_context, string &dumpJson);

Result LoadModelFromFile(string &modelFile, uint32_t *modelId);

Result Execute(uint32_t modelId, aclmdlDataset *input, aclmdlDataset *output, string outputPrefix, uint32_t nodeId,
               aclmdlDesc *modelDesc, vector <aclrtContext> &v_context, uint32_t remoteDevice, uint32_t loopNum,
               string dynamicHWSize, uint32_t dynamicBatch, string dynamicDims, uint32_t batchSize,
               double &totalSamplesTime);

Result WriteResult(string modelFile, string outputPath, uint32_t samplesNumber, uint32_t stepNumber,
                   double totalAverageTime, double totalAverageFPS);

#endif //RUN_ACL_MODEL_UTILS_H
