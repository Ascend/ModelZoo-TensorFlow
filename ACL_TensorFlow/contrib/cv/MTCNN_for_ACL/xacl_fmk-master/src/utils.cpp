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

#include "utils.h"

using namespace std;

void Help() {
    printf("%s - I - [XACL]: Usage: ./xacl_fmk [input parameters]\n"
           "-m=model                 Required, om model file path\n"
           "                         Relative and absolute paths are supported\n"
           "-o=outputPrefix          Required, prefix of _output_{outputIdx}_{loopIdx}.bin\n"
           "                         Relative and absolute paths are supported\n"
           "\n"
           "-i=inputFiles            Optional, input bin files or input file directories, use commas (,) to separate multiple inputs\n"
           "                         Relative and absolute paths are supported, set inputs to all zeros if not specified\n"
           "-g=mergeInput            Optional, whether merge input by batch size, only take effect in directories input\n"
           "                         The default value is false, in this case, each input must be saved in N batches\n"
           "                         Otherwise, each input must be saved in 1 batch and will be merged to N batches automatically\n"
           "-d=dumpJson              Optional, Configuration file used to save operator input and output data\n"
           "                         The default value is NULL, indicating that operator input and output data is not saved\n"
           "-n=nodeId                Optional, ID of the NPU used for inference\n"
           "                         The default value is 0, indicating that device 0 is used for inference\n"
           "-l=loopNum               Optional, The number of inference times\n"
           "                         The default value is 1, indicating that inference is performed once\n"
           "-b=batchSize             Optional, Size of the static batch\n"
           "                         The default value is 1, indicating that the static batch is 1\n"
           "                         Static batch will be disabled when dynamic batch has been set\n"
           "-v=dynamicShape          Optional, Size of the dynamic shape\n"
           "                         Use semicolon (;) to separate each input, use commas (,) to separate each dim\n"
           "                         The default value is NULL, indicating that the dynamicShape function is disabled\n"
           "                         Enter the actual shape size when the dynamicShape function is enabled\n"
           "-w=dynamicOutSize        Optional, Size of the output memory\n"
           "                         Use semicolon (;) to separate each output\n"
           "                         The default value is NULL, indicating that the dynamicShape function is disabled\n"
           "                         Enter the actual output size when the dynamicShape function is enabled\n"
           "-x=dynamicHWSize         Optional, Size of the dynamic height and width, use commas (,) to separate\n"
           "                         The default value is NULL, indicating that the dynamicHW function is disabled\n"
           "                         Enter the actual height and width size when the dynamicHW function is enabled\n"
           "-y=dynamicBatch          Optional, Size of the dynamic batch, cannot be used with dynamicHeight or dynamicWidth\n"
           "                         The default value is 0, indicating that the dynamic batch function is disabled\n"
           "                         Enter the actual size of the batch when the dynamic batch function is enabled\n"
           "-z=dynamicDims           Optional, Size of the dynamicDims, use commas (,) to separate\n"
           "                         The default value is NULL, indicating that the dynamicDims function is disabled\n"
           "                         Enter the actual size of each dims when the dynamicDims function is enabled\n"
           "-r=remoteDevice          Optional, Whether the NPU is deployed remotely\n"
           "                         The default value is 0, indicating that the NPU is co-deployed as 1951DC\n"
           "                         The value 1 indicates that the NPU is deployed remotely as 1951MDC\n"
           "\n"
           "-h=help                  Show this help message\n", GetTime().c_str());
}

string GetTime() {
    struct timeval timeEval;
    gettimeofday(&timeEval, NULL);
    int milliSecond = timeEval.tv_usec / 1000;

    time_t timeStamp;
    time(&timeStamp);
    char secondTime[20];
    strftime(secondTime, sizeof(secondTime), "%Y-%m-%d %H:%M:%S", localtime(&timeStamp));

    char milliTime[24];
    snprintf(milliTime, sizeof(milliTime), "%s.%03d", secondTime, milliSecond);

    return milliTime;
}

string FormatInt(string Idx, size_t formatSize) {
    size_t sizeIdx = Idx.size();
    if (sizeIdx < formatSize) {
        for (uint32_t i = 0; i < formatSize - sizeIdx; ++i) {
            Idx = "0" + Idx;
        }
    }
    return Idx;
}

Result CreateDir(string pathName) {
    if (access(pathName.c_str(), F_OK) == -1) {
        printf("%s - I - [XACL]: Path %s is not exist, try to create it\n",
               GetTime().c_str(), pathName.c_str());

        if (mkdir(pathName.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH)) {
            printf("%s - E - [XACL]: Create path failed\n",
                   GetTime().c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

void Split(const string &s, vector <string> &tokens, const string &delimiters = ";") {
    string::size_type lastPos = s.find_first_not_of(delimiters, 0);
    string::size_type pos = s.find_first_of(delimiters, lastPos);
    while (string::npos != pos || string::npos != lastPos) {
        tokens.push_back(s.substr(lastPos, pos - lastPos));
        lastPos = s.find_first_not_of(delimiters, pos);
        pos = s.find_first_of(delimiters, lastPos);
    }
}

int ScanFiles(vector <string> &fileList, string inputDirectory) {
    const char *str = inputDirectory.c_str();
    DIR *dir = opendir(str);
    struct dirent *p = NULL;
    while ((p = readdir(dir)) != NULL) {
        if (p->d_name[0] != '.') {
            string name = string(p->d_name);
            fileList.push_back(inputDirectory + '/' + name);
        }
    }
    closedir(dir);

    if (fileList.size() == 0) {
        printf("%s - E - [XACL]: No file in the directory: %s\n",
               GetTime().c_str(), str);
    }

    return fileList.size();
}

void MergeInputFile(vector <string> &fileList, string &outputFile) {
    ofstream fileOut(outputFile, ofstream::binary);
    for (uint32_t fileIdx = 0; fileIdx < fileList.size(); ++fileIdx) {
        ifstream binFile(fileList[fileIdx], ifstream::binary);

        binFile.seekg(0, binFile.beg);
        while (!binFile.eof()) {
            char szBuf[256] = {'\0'};
            binFile.read(szBuf, sizeof(char) * 256);
            int length = binFile.gcount();
            fileOut.write(szBuf, length);
        }
        binFile.close();
    }
    fileOut.close();
}

void ModelInputRelease(aclmdlDataset *input) {
    aclError ret = ACL_ERROR_NONE;
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(input, i);
        void *data = aclGetDataBufferAddr(dataBuffer);

        ret = aclrtFree(data);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclrtFree return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
        }

        ret = aclDestroyDataBuffer(dataBuffer);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclDestroyDataBuffer return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
        }
    }

    ret = aclmdlDestroyDataset(input);
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclmdlDestroyDataset return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
    }
    printf("%s - I - [XACL]: Destroy input data success\n",
           GetTime().c_str());
}

void ModelOutputRelease(aclmdlDataset *output) {
    aclError ret = ACL_ERROR_NONE;
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(output, i);
        void *data = aclGetDataBufferAddr(dataBuffer);

        ret = aclrtFree(data);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclrtFree return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
        }

        ret = aclDestroyDataBuffer(dataBuffer);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclDestroyDataBuffer return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
        }
    }

    ret = aclmdlDestroyDataset(output);
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclmdlDestroyDataset return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
    }
    printf("%s - I - [XACL]: Destroy output data success\n",
           GetTime().c_str());
}

void ModelUnloadAndDescDestroy(uint32_t modelId, aclmdlDesc *modelDesc) {
    aclError ret = ACL_ERROR_NONE;
    ret = aclmdlUnload(modelId);
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclmdlUnload return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
    }

    ret = aclmdlDestroyDesc(modelDesc);
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclmdlDestroyDesc return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
    }
}

void DeviceContextDestroy(uint32_t nodeId, vector <aclrtContext> &v_context) {
    aclError ret = ACL_ERROR_NONE;
    for (auto iter = v_context.begin(); iter != v_context.end(); iter++) {
        ret = aclrtDestroyContext(*iter);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclrtDestroyContext return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
        }
    }

    ret = aclrtResetDevice(nodeId);
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclrtResetDevice return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
    }

    printf("%s - I - [XACL]: Start to finalize acl, aclFinalize interface adds 2s delay to upload device logs\n",
           GetTime().c_str());
    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclFinalize return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
    }

    printf("%s - I - [XACL]: Finalize acl success\n",
           GetTime().c_str());
}

void ReleaseAllModelResource(uint32_t modelId, uint32_t nodeId,
                             aclmdlDesc *modelDesc, vector <aclrtContext> &v_context) {
    if (modelDesc && modelId) {
        ModelUnloadAndDescDestroy(modelId, modelDesc);
    }
    DeviceContextDestroy(nodeId, v_context);
}

void *ReadBinFile(string fileName, uint32_t &fileSize, uint32_t remoteDevice) {
    ifstream binFile(fileName, ifstream::binary);
    if (binFile.is_open() == false) {
        printf("%s - E - [XACL]: Open input file failed\n",
               GetTime().c_str());
        return nullptr;
    }

    binFile.seekg(0, binFile.end);
    uint32_t binFileBufferLen = binFile.tellg();
    binFile.seekg(0, binFile.beg);

    aclError ret = ACL_ERROR_NONE;
    void *binFileBufferData = nullptr;
    if (remoteDevice == 0) {
        ret = aclrtMallocHost(&binFileBufferData, binFileBufferLen);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclrtMallocHost return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
            binFile.close();
            return nullptr;
        }
    } else {
        ret = aclrtMalloc(&binFileBufferData, binFileBufferLen, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclrtMalloc return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
            binFile.close();
            return nullptr;
        }
    }

    binFile.read(static_cast<char *>(binFileBufferData), binFileBufferLen);
    binFile.close();
    fileSize = binFileBufferLen;

    return binFileBufferData;
}

void *GetDeviceBufferFromFile(string fileName, uint32_t &fileSize, uint32_t remoteDevice) {
    uint32_t inputHostBuffSize = 0;
    void *inputHostBuff = ReadBinFile(fileName, inputHostBuffSize, remoteDevice);
    if (inputHostBuff == nullptr) {
        return nullptr;
    }

    if (remoteDevice == 0) {
        aclError ret = ACL_ERROR_NONE;
        void *inBufferDev = nullptr;
        uint32_t inBufferSize = inputHostBuffSize;
        ret = aclrtMalloc(&inBufferDev, inBufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclrtMalloc return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
            ret = aclrtFreeHost(inputHostBuff);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclrtFreeHost return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
            }

            return nullptr;
        }

        ret = aclrtMemcpy(inBufferDev, inBufferSize, inputHostBuff, inputHostBuffSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclrtMemcpy return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());

            ret = aclrtFree(inBufferDev);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclrtFree return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
            }

            ret = aclrtFreeHost(inputHostBuff);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclrtFreeHost return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
            }

            return nullptr;
        }
        ret = aclrtFreeHost(inputHostBuff);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclrtFreeHost return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
        }
        fileSize = inBufferSize;
        return inBufferDev;
    } else {
        fileSize = inputHostBuffSize;
        return inputHostBuff;
    }
}

Result InitResource(uint32_t nodeId, vector <aclrtContext> &v_context, string &dumpJson) {
    aclError ret = ACL_ERROR_NONE;
    ret = aclInit((char *) dumpJson.c_str());
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclInit return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
        return FAILED;
    }

    ret = aclrtSetDevice(nodeId);
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclrtSetDevice return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
        return FAILED;
    }

    aclrtContext context;
    ret = aclrtCreateContext(&context, nodeId);
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclrtCreateContext return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
        return FAILED;
    }
    v_context.push_back(context);

    return SUCCESS;
}

Result LoadModelFromFile(string &modelFile, uint32_t *modelId) {
    aclError ret = ACL_ERROR_NONE;
    ret = aclmdlLoadFromFile(modelFile.c_str(), modelId);
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclmdlLoadFromFile return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
        return FAILED;
    }
    return SUCCESS;
}

aclmdlDesc *CreateDesc(aclmdlDesc *modelDesc, uint32_t modelId) {
    modelDesc = aclmdlCreateDesc();
    if (modelDesc == nullptr) {
        printf("%s - E - [XACL]: Interface of aclmdlCreateDesc return failed\n",
               GetTime().c_str());
        return nullptr;
    }

    aclError ret = ACL_ERROR_NONE;
    ret = aclmdlGetDesc(modelDesc, modelId);
    if (ret != ACL_ERROR_NONE) {
        printf("%s - E - [XACL]: Interface of aclmdlGetDesc return failed, error message is:\n%s\n",
               GetTime().c_str(), aclGetRecentErrMsg());
        return nullptr;
    }

    return modelDesc;
}

aclmdlDataset *CreateInput(aclmdlDesc *modelDesc, aclmdlDataset *input,
                           vector <string> v_inputFiles, uint32_t remoteDevice) {
    input = aclmdlCreateDataset();
    if (input == nullptr) {
        printf("%s - E - [XACL]: Interface of aclmdlCreateDataset return failed\n",
               GetTime().c_str());
        return nullptr;
    }

    size_t inputSize = aclmdlGetNumInputs(modelDesc);
    if (inputSize != v_inputFiles.size()) {
        printf("%s - E - [XACL]: Input file number not match, [%zu / %zu]\n",
               GetTime().c_str(), v_inputFiles.size(), inputSize);
        return nullptr;
    }

    vector<void *> inputBuffer(v_inputFiles.size(), nullptr);
    for (size_t inIdx = 0; inIdx < inputSize; ++inIdx) {
        uint32_t bufferSize;
        inputBuffer[inIdx] = GetDeviceBufferFromFile(v_inputFiles[inIdx], bufferSize, remoteDevice);
        aclDataBuffer *inputData = aclCreateDataBuffer((void *) (inputBuffer[inIdx]), bufferSize);
        if (inputData == nullptr) {
            printf("%s - E - [XACL]: Interface of aclCreateDataBuffer return failed\n",
                   GetTime().c_str());
            ModelInputRelease(input);
            return nullptr;
        }

        aclError ret = ACL_ERROR_NONE;
        ret = aclmdlAddDatasetBuffer(input, inputData);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclmdlAddDatasetBuffer return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
            ModelInputRelease(input);
            inputData = nullptr;
            return nullptr;
        }
        printf("%s - I - [XACL]: The buffer size of input %zu: %d\n",
               GetTime().c_str(), inIdx, bufferSize);
    }

    return input;
}

Result CreateInputDesc(aclmdlDataset *input, vector <string> v_inputFiles, string dynamicShape,
                       aclTensorDesc *inputDesc[]) {
    vector <string> v_dynamicShape;
    Split(dynamicShape, v_dynamicShape, ";");
    if (v_dynamicShape.size() != v_inputFiles.size()) {
        printf("%s - I - [XACL]: dynamicShape input numbers %zu are not equal with actually input numbers %zu\n",
               GetTime().c_str(), v_dynamicShape.size(), v_inputFiles.size());
        ModelInputRelease(input);
        return FAILED;
    }

    for (uint32_t inIdx = 0; inIdx < v_dynamicShape.size(); ++inIdx) {
        aclError ret = ACL_ERROR_NONE;
        vector <string> v_shapes;
        Split(v_dynamicShape[inIdx], v_shapes, ",");
        int64_t shapes[v_shapes.size()];
        for (uint32_t dimIdx = 0; dimIdx < v_shapes.size(); ++dimIdx) {
            printf("%s - I - [XACL]: The %d input dynamicShape index %d is: %s\n",
                   GetTime().c_str(), inIdx, dimIdx, v_shapes[dimIdx].c_str());
            shapes[dimIdx] = atoi(v_shapes[dimIdx].c_str());
        }
        //ACL_FLOAT16 与 ACL_FORMAT_NHWC随意填写，暂时不生效
        inputDesc[inIdx] = aclCreateTensorDesc(ACL_FLOAT16, v_shapes.size(), shapes, ACL_FORMAT_NHWC);

        ret = aclmdlSetDatasetTensorDesc(input, inputDesc[inIdx], inIdx);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - I - [XACL]: Interface of aclmdlSetDatasetTensorDesc return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
            ModelInputRelease(input);
            return FAILED;
        }
    }
    return SUCCESS;
}

aclmdlDataset *CreateZeroInput(aclmdlDesc *modelDesc, aclmdlDataset *input,
                               vector <string> v_inputFiles, uint32_t remoteDevice) {
    input = aclmdlCreateDataset();
    if (input == nullptr) {
        printf("%s - E - [XACL]: Interface of aclmdlCreateDataset return failed\n",
               GetTime().c_str());
        return nullptr;
    }

    size_t inputSize = aclmdlGetNumInputs(modelDesc);
    printf("%s - I - [XACL]: The number of inputs queried through aclmdlGetNumInputs is: %zu\n",
           GetTime().c_str(), inputSize);

    for (size_t i = 0; i < inputSize; i++) {
        size_t bufferSizeZero = aclmdlGetInputSizeByIndex(modelDesc, i);
        printf("%s - I - [XACL]: The buffer size of input %zu: %zu\n",
               GetTime().c_str(), i, bufferSizeZero);

        void *inputBuffer = nullptr;
        if (!remoteDevice) {
            void *binFileBufferData = nullptr;
            aclError ret = ACL_ERROR_NONE;
            ret = aclrtMallocHost(&binFileBufferData, bufferSizeZero);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclrtMallocHost return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
                return nullptr;
            }

            memset(binFileBufferData, 0, bufferSizeZero);

            ret = aclrtMalloc(&inputBuffer, bufferSizeZero, ACL_MEM_MALLOC_NORMAL_ONLY);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclrtMalloc return failed, size is %zu, error message is:\n%s\n",
                       GetTime().c_str(), bufferSizeZero, aclGetRecentErrMsg());
                return nullptr;
            }

            ret = aclrtMemcpy(inputBuffer, bufferSizeZero, binFileBufferData, bufferSizeZero,
                              ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclrtMemcpy return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
                ModelInputRelease(input);
                return nullptr;
            }

            ret = aclrtFreeHost(binFileBufferData);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclrtFreeHost return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
            }

        } else {
            aclError ret = ACL_ERROR_NONE;
            ret = aclrtMalloc(&inputBuffer, bufferSizeZero, ACL_MEM_MALLOC_NORMAL_ONLY);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclrtMalloc return failed, size is %zu, error message is:\n%s\n",
                       GetTime().c_str(), bufferSizeZero, aclGetRecentErrMsg());
                return nullptr;
            }
            memset(inputBuffer, 0, bufferSizeZero);
        }

        aclDataBuffer *inputData = aclCreateDataBuffer(inputBuffer, bufferSizeZero);
        if (inputData == nullptr) {
            printf("%s - E - [XACL]: Interface of aclCreateDataBuffer return failed\n",
                   GetTime().c_str());
            ModelInputRelease(input);
            return nullptr;
        }

        aclError ret = ACL_ERROR_NONE;
        ret = aclmdlAddDatasetBuffer(input, inputData);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclmdlAddDatasetBuffer return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
            ModelInputRelease(input);
            inputData = nullptr;
            return nullptr;
        }
    }
    return input;
}

aclmdlDataset *CreateOutput(aclmdlDesc *modelDesc, aclmdlDataset *output, aclmdlDataset *input, string dynamicOutSize) {
    if (modelDesc == nullptr) {
        printf("%s - E - [XACL]: No model desc, create output failed\n",
               GetTime().c_str());
        return nullptr;
    }

    output = aclmdlCreateDataset();
    if (output == nullptr) {
        printf("%s - E - [XACL]: Interface of aclmdlCreateDataset return failed\n",
               GetTime().c_str());
        ModelInputRelease(input);
        return nullptr;
    }

    size_t outputSize = aclmdlGetNumOutputs(modelDesc);
    for (size_t outIdx = 0; outIdx < outputSize; ++outIdx) {
        size_t bufferSize = aclmdlGetOutputSizeByIndex(modelDesc, outIdx);
        if (bufferSize == 0) {
            if (dynamicOutSize == "") {
                printf("%s - I - [XACL]: Output size is zero, malloc 1 byte\n",
                       GetTime().c_str());
                bufferSize = 1;
            } else {
                vector <string> v_dynamicOutSize;
                Split(dynamicOutSize, v_dynamicOutSize, ";");
                bufferSize = atoi(v_dynamicOutSize[outIdx].c_str());
                printf("%s - I - [XACL]: Dynamic output, set size to %zu\n",
                       GetTime().c_str(), bufferSize);
            }
        }

        void *outputBuffer = nullptr;

        aclError ret = ACL_ERROR_NONE;
        ret = aclrtMalloc(&outputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclrtMalloc return failed, size is %zu, error message is:\n%s\n",
                   GetTime().c_str(), bufferSize, aclGetRecentErrMsg());
            ModelInputRelease(input);
            ModelOutputRelease(output);
            return nullptr;
        }

        aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer, bufferSize);
        if (outputData == nullptr) {
            printf("%s - E - [XACL]: Interface of aclCreateDataBuffer return failed\n",
                   GetTime().c_str());
            ModelInputRelease(input);
            ModelOutputRelease(output);
            return nullptr;
        }

        ret = aclmdlAddDatasetBuffer(output, outputData);
        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclmdlAddDatasetBuffer return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());
            ModelInputRelease(input);
            ModelOutputRelease(output);
            return nullptr;
        }
    }
    return output;
}

Result Execute(uint32_t modelId, aclmdlDataset *input, aclmdlDataset *output, string outputPrefix, uint32_t nodeId,
               aclmdlDesc *modelDesc, vector <aclrtContext> &v_context, uint32_t remoteDevice, uint32_t loopNum,
               string dynamicHWSize, uint32_t dynamicBatch, string dynamicDims, uint32_t batchSize,
               double &totalSamplesTime) {
    // 拆分 dynamicHWSize
    uint32_t dynamicHeight = 0;
    uint32_t dynamicWidth = 0;
    if (dynamicHWSize != "") {
        vector <string> v_dynamicHWSize;
        Split(dynamicHWSize, v_dynamicHWSize, ",");
        for (uint32_t Idx = 0; Idx < v_dynamicHWSize.size(); ++Idx) {
            if (Idx == 0) {
                printf("%s - I - [XACL]: dynamicHeight is: %s\n",
                       GetTime().c_str(), v_dynamicHWSize[Idx].c_str());
                dynamicHeight = atoi(v_dynamicHWSize[Idx].c_str());
            } else if (Idx == 1) {
                printf("%s - I - [XACL]: dynamicWidth is: %s\n",
                       GetTime().c_str(), v_dynamicHWSize[Idx].c_str());
                dynamicWidth = atoi(v_dynamicHWSize[Idx].c_str());
            } else {
                printf("%s - E - [XACL]: dynamicHWSize only has two members which represent the height and width\n",
                       GetTime().c_str());
                ModelInputRelease(input);
                ModelOutputRelease(output);
                return FAILED;
            }
        }

        // 动态HW时，必须同时提供H和W大小
        if ((dynamicHeight != 0 && dynamicWidth == 0) || (dynamicHeight == 0 && dynamicWidth != 0)) {
            printf("%s - E - [XACL]: Both height and width are needed when enable dynamicHWSize\n",
                   GetTime().c_str());
            ModelInputRelease(input);
            ModelOutputRelease(output);
            return FAILED;
        }
    }

    // 拆分 dynamicDims
    uint32_t dimCount = 0;
    aclmdlIODims currentDims;
    if (dynamicDims != "") {
        vector <string> v_dynamicDims;
        Split(dynamicDims, v_dynamicDims, ",");
        for (uint32_t Idx = 0; Idx < v_dynamicDims.size(); ++Idx) {
            printf("%s - I - [XACL]: dynamicDims index %d is: %s\n",
                   GetTime().c_str(), Idx, v_dynamicDims[Idx].c_str());
            currentDims.dims[Idx] = atoi(v_dynamicDims[Idx].c_str());
            dimCount++;
        }
        printf("%s - I - [XACL]: dynamicDims size is: %d\n",
               GetTime().c_str(), dimCount);
        currentDims.dimCount = dimCount;
    }

    struct timeval startTimeStamp;
    struct timeval endTimeStamp;
    double totalTime = 0;
    double costTime;
    double startTime;
    double endTime;
    for (uint32_t loopIdx = 0; loopIdx < loopNum; ++loopIdx) {
        aclError ret = ACL_ERROR_NONE;

        if (dynamicBatch > 0) {
            size_t dynamicIndex;
            ret = aclmdlGetInputIndexByName(modelDesc, ACL_DYNAMIC_TENSOR_NAME, &dynamicIndex);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclmdlGetInputIndexByName return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
                ModelInputRelease(input);
                ModelOutputRelease(output);
                return FAILED;
            }

            ret = aclmdlSetDynamicBatchSize(modelId, input, dynamicIndex, dynamicBatch);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclmdlSetDynamicBatchSize return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
                ModelInputRelease(input);
                ModelOutputRelease(output);
                return FAILED;
            }

            // 动态Batch时将实际batch赋值为动态Batch大小
            printf("%s - I - [XACL]: Due to dynamicBatchSize is set, static batch size resets to the same value\n",
                   GetTime().c_str());
            batchSize = dynamicBatch;
            printf("%s - I - [XACL]: Static batch size resets to %d\n",
                   GetTime().c_str(), batchSize);
        }

        if (dynamicHeight > 0 && dynamicWidth > 0) {
            size_t dynamicIndex;
            ret = aclmdlGetInputIndexByName(modelDesc, ACL_DYNAMIC_TENSOR_NAME, &dynamicIndex);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclmdlGetInputIndexByName return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
                ModelInputRelease(input);
                ModelOutputRelease(output);
                return FAILED;
            }

            ret = aclmdlSetDynamicHWSize(modelId, input, dynamicIndex, dynamicHeight, dynamicWidth);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclmdlSetDynamicHWSize return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
                ModelInputRelease(input);
                ModelOutputRelease(output);
                return FAILED;
            }
        }

        if (dimCount > 0) {
            size_t dynamicIndex;
            ret = aclmdlGetInputIndexByName(modelDesc, ACL_DYNAMIC_TENSOR_NAME, &dynamicIndex);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclmdlGetInputIndexByName return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
                ModelInputRelease(input);
                ModelOutputRelease(output);
                return FAILED;
            }

            ret = aclmdlSetInputDynamicDims(modelId, input, dynamicIndex, &currentDims);
            if (ret != ACL_ERROR_NONE) {
                printf("%s - E - [XACL]: Interface of aclmdlSetInputDynamicDims return failed, error message is:\n%s\n",
                       GetTime().c_str(), aclGetRecentErrMsg());
                ModelInputRelease(input);
                ModelOutputRelease(output);
                return FAILED;
            }
        }

        gettimeofday(&startTimeStamp, NULL);

        ret = aclmdlExecute(modelId, input, output);

        gettimeofday(&endTimeStamp, NULL);

        if (ret != ACL_ERROR_NONE) {
            printf("%s - E - [XACL]: Interface of aclmdlExecute return failed, error message is:\n%s\n",
                   GetTime().c_str(), aclGetRecentErrMsg());

            printf("%s - E - [XACL]: Run acl model failed\n",
                   GetTime().c_str());
            ModelInputRelease(input);
            ModelOutputRelease(output);
            return FAILED;
        } else {
            printf("%s - I - [XACL]: Run acl model success\n",
                   GetTime().c_str());
        }

        costTime = (1.0 * (endTimeStamp.tv_sec - startTimeStamp.tv_sec) * 1000000 +
                    (endTimeStamp.tv_usec - startTimeStamp.tv_usec)) / 1000;
        startTime = (1.0 * startTimeStamp.tv_sec * 1000000 + startTimeStamp.tv_usec) / 1000;
        endTime = (1.0 * endTimeStamp.tv_sec * 1000000 + endTimeStamp.tv_usec) / 1000;
        totalTime += costTime;
        printf("%s - I - [XACL]: Loop %d, start timestamp %4.0f, end timestamp %4.0f, cost time %4.2fms\n",
               GetTime().c_str(), loopIdx, startTime, endTime, costTime);

        string outputBinFileName(outputPrefix + "_output_");
        string loopName = "_" + FormatInt(to_string(loopIdx), 3);
        for (size_t outIndex = 0; outIndex < aclmdlGetDatasetNumBuffers(output); ++outIndex) {
            string outputBinFileNameIdx = outputBinFileName + FormatInt(to_string(outIndex), 2) + loopName + ".bin";
            FILE *fop = fopen(outputBinFileNameIdx.c_str(), "wb+");
            aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(output, outIndex);
            void *data = aclGetDataBufferAddr(dataBuffer);
            uint32_t len = aclGetDataBufferSizeV2(dataBuffer);
            if (remoteDevice == 0) {
                void *outHostData = NULL;
                ret = aclrtMallocHost(&outHostData, len);
                if (ret != ACL_ERROR_NONE) {
                    printf("%s - E - [XACL]: Interface of aclrtMallocHost return failed, error message is:\n%s\n",
                           GetTime().c_str(), aclGetRecentErrMsg());
                    ModelInputRelease(input);
                    ModelOutputRelease(output);
                    return FAILED;
                }
                ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
                if (ret != ACL_ERROR_NONE) {
                    printf("%s - E - [XACL]: Interface of aclrtMemcpy return failed, error message is:\n%s\n",
                           GetTime().c_str(), aclGetRecentErrMsg());
                    ModelInputRelease(input);
                    ModelOutputRelease(output);
                    return FAILED;
                }

                size_t len1 = fwrite(outHostData, sizeof(char), len, fop);
                if (len1 != len) {
                    printf("%s - E - [XACL]: Write output bin file failed\n",
                           GetTime().c_str());
                }
                fclose(fop);

                ret = aclrtFreeHost(outHostData);
                if (ret != ACL_ERROR_NONE) {
                    printf("%s - E - [XACL]: Interface of aclrtFreeHost return failed, error message is:\n%s\n",
                           GetTime().c_str(), aclGetRecentErrMsg());
                    ModelInputRelease(input);
                    ModelOutputRelease(output);
                    return FAILED;
                }
            } else {
                void *outHostData = NULL;
                ret = aclrtMalloc(&outHostData, len, ACL_MEM_MALLOC_NORMAL_ONLY);
                if (ret != ACL_ERROR_NONE) {
                    printf("%s - E - [XACL]: Interface of aclrtMalloc return failed, error message is:\n%s\n",
                           GetTime().c_str(), aclGetRecentErrMsg());
                    ModelInputRelease(input);
                    ModelOutputRelease(output);
                    return FAILED;
                }

                ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_DEVICE);
                if (ret != ACL_ERROR_NONE) {
                    printf("%s - E - [XACL]: Interface of aclrtMemcpy return failed, error message is:\n%s\n",
                           GetTime().c_str(), aclGetRecentErrMsg());
                    ModelInputRelease(input);
                    ModelOutputRelease(output);
                    return FAILED;
                }

                size_t len1 = fwrite(outHostData, sizeof(char), len, fop);
                if (len1 != len) {
                    printf("%s - E - [XACL]: Dump output to file return failed\n",
                           GetTime().c_str());
                }
                fclose(fop);

                ret = aclrtFree(outHostData);
                if (ret != ACL_ERROR_NONE) {
                    printf("%s - E - [XACL]: Interface of aclrtFree return failed, error message is:\n%s\n",
                           GetTime().c_str(), aclGetRecentErrMsg());
                    ModelInputRelease(input);
                    ModelOutputRelease(output);
                    return FAILED;
                }
            }
            printf("%s - I - [XACL]: Dump output %ld to file success\n",
                   GetTime().c_str(), outIndex);
        }
    }

    totalSamplesTime = totalSamplesTime + (totalTime / loopNum);

    printf("%s - I - [XACL]: Single step average NPU inference time of %d loops: %f ms %4.2f fps\n",
           GetTime().c_str(), loopNum, (totalTime / loopNum), (1000 * loopNum * batchSize / totalTime));

    ModelInputRelease(input);
    ModelOutputRelease(output);

    return SUCCESS;
}

Result WriteResult(string modelFile, string outputPath, uint32_t samplesNumber, uint32_t stepNumber,
                   double totalAverageTime, double totalAverageFPS) {
    vector <string> v_modelFile;
    Split(modelFile, v_modelFile, "/");
    string modelName = v_modelFile[v_modelFile.size() - 1];
    vector <string> v_modelName;
    Split(modelName, v_modelName, ".");

    string resultFileName = outputPath + v_modelName[0] + "_performance.txt";

    ofstream resultFile(resultFileName.c_str(), ofstream::out);

    if (!resultFile) {
        printf("%s - I - [XACL]: Open acl result file failed\n",
               GetTime().c_str());
        return FAILED;
    } else {
        printf("%s - I - [XACL]: Write acl result to file %s\n",
               GetTime().c_str(), resultFileName.c_str());
    }

    resultFile << samplesNumber << " samples average NPU inference time of " << stepNumber << " steps: ";
    resultFile << totalAverageTime << "ms " << totalAverageFPS << " fps" << endl;
    resultFile.close();

    return SUCCESS;
}
