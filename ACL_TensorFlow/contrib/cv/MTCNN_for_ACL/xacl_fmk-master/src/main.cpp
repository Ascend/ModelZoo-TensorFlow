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

int main(int argc, char **argv) {
    // 入参列表
    string inputFiles = "";
    string mergeInput = "";
    string outputPrefix = "";
    string modelFile = "";
    string dumpJson = "";
    string dynamicShape = "";
    string dynamicOutSize = "";
    string dynamicHWSize = "";
    string dynamicDims = "";
    uint32_t dynamicBatch = 0;
    uint32_t modelId = 0;
    uint32_t nodeId = 0;
    uint32_t loopNum = 1;
    uint32_t batchSize = 1;
    uint32_t remoteDevice = 0;

    // 获取入参
    while (1) {
        int optionIdx = 0;
        int optionInput;
        struct option longOptions[] = {
                {"inputFiles",     1, 0, 'i'},
                {"mergeInput",     1, 0, 'g'},
                {"outputPrefix",   1, 0, 'o'},
                {"modelFile",      1, 0, 'm'},
                {"nodeId",         1, 0, 'n'},
                {"loopNum",        1, 0, 'l'},
                {"dumpJson",       1, 0, 'd'},
                {"batchSize",      1, 0, 'b'},
                {"dynamicShape",   1, 0, 'v'},
                {"dynamicOutSize", 1, 0, 'w'},
                {"dynamicHWSize",  1, 0, 'x'},
                {"dynamicBatch",   1, 0, 'y'},
                {"dynamicDims",    1, 0, 'z'},
                {"remoteDevice",   1, 0, 'r'},
                {"help",           0, 0, 'h'},
        };

        optionInput = getopt_long(argc, argv, "i:g:o:m:n:l:d:b:v:w:x:y:z:r:h", longOptions, &optionIdx);
        if (optionInput == -1) {
            break;
        }
        switch (optionInput) {
            case 'i': {
                inputFiles = string(optarg);
                printf("%s - I - [XACL]: Input files: %s\n",
                       GetTime().c_str(), inputFiles.c_str());
                break;
            }

            case 'g': {
                mergeInput = string(optarg);
                transform(mergeInput.begin(), mergeInput.end(), mergeInput.begin(), ::tolower);
                printf("%s - I - [XACL]: Merge input flag: %s\n",
                       GetTime().c_str(), mergeInput.c_str());
                break;
            }

            case 'o': {
                outputPrefix = string(optarg);
                printf("%s - I - [XACL]: Output file prefix: %s\n",
                       GetTime().c_str(), outputPrefix.c_str());
                break;
            }

            case 'm': {
                modelFile = string(optarg);
                printf("%s - I - [XACL]: NPU model file: %s\n",
                       GetTime().c_str(), modelFile.c_str());
                break;
            }

            case 'n': {
                nodeId = atoi(optarg);
                printf("%s - I - [XACL]: NPU device index: %d\n",
                       GetTime().c_str(), nodeId);
                break;
            }

            case 'l': {
                loopNum = atoi(optarg);
                printf("%s - I - [XACL]: Execution loops: %d\n",
                       GetTime().c_str(), loopNum);
                break;
            }

            case 'd': {
                dumpJson = string(optarg);
                printf("%s - I - [XACL]: Dump config file: %s\n",
                       GetTime().c_str(), dumpJson.c_str());
                break;
            }

            case 'b': {
                batchSize = atoi(optarg);
                printf("%s - I - [XACL]: Static batch size: %d\n",
                       GetTime().c_str(), batchSize);
                break;
            }

            case 'v': {
                dynamicShape = string(optarg);
                printf("%s - I - [XACL]: Dynamic shape size: %s\n",
                       GetTime().c_str(), dynamicShape.c_str());
                break;
            }

            case 'w': {
                dynamicOutSize = string(optarg);
                printf("%s - I - [XACL]: Dynamic output memory size: %s\n",
                       GetTime().c_str(), dynamicOutSize.c_str());
                break;
            }

            case 'x': {
                dynamicHWSize = string(optarg);
                printf("%s - I - [XACL]: Dynamic HW size: %s\n",
                       GetTime().c_str(), dynamicHWSize.c_str());
                break;
            }

            case 'y': {
                dynamicBatch = atoi(optarg);
                printf("%s - I - [XACL]: Dynamic batch size: %d\n",
                       GetTime().c_str(), dynamicBatch);
                break;
            }

            case 'z': {
                dynamicDims = string(optarg);
                printf("%s - I - [XACL]: Dynamic dims size: %s\n",
                       GetTime().c_str(), dynamicDims.c_str());
                break;
            }

            case 'r': {
                remoteDevice = atoi(optarg);
                printf("%s - I - [XACL]: Remote device flag: %d\n",
                       GetTime().c_str(), remoteDevice);
                break;
            }

            case 'h': {
                Help();
                return SUCCESS;
            }
        }
    }
    Result ret;
    // 判断输出文件以及模型文件是否存在，不存在则报错退出
    if (outputPrefix == "" || modelFile == "") {
        printf("%s - E - [XACL]: Output file prefix (-o) and npu model file (-m) parameters are required\n",
               GetTime().c_str());
        return FAILED;
    }

    // 判断output目录是否存在，若不存在则创建目录，创建失败则退出
    string outputPath = "";
    vector <string> v_outputPrefix;
    if (outputPrefix.find("./") == 0) {
        // 判断output目录是否是以./开头的相对路径
        outputPath = "./";
    } else if (outputPrefix.find("/") == 0) {
        // 判断output目录是绝对路径
        outputPath = "/";
    }
    Split(outputPrefix, v_outputPrefix, "/");

    if (v_outputPrefix.size() != 1) {
        for (uint32_t i = 0; i < v_outputPrefix.size() - 1; i++) {
            if (v_outputPrefix[i] == ".") continue;
            outputPath += v_outputPrefix[i] + "/";
            ret = CreateDir(outputPath);
            if (ret != SUCCESS) {
                return FAILED;
            }
        }
        printf("%s - I - [XACL]: Output path is: %s\n",
               GetTime().c_str(), outputPath.c_str());
    } else {
        printf("%s - I - [XACL]: Output path is: ./\n",
               GetTime().c_str());
    }

    // 动态HW、动态Batch和动态Dims分档以及动态Shape不能同时存在
    if ((dynamicBatch != 0 && dynamicHWSize != "") ||
        (dynamicBatch != 0 && dynamicDims != "") ||
        (dynamicBatch != 0 && dynamicShape != "") ||
        (dynamicHWSize != "" && dynamicDims != "") ||
        (dynamicHWSize != "" && dynamicShape != "") ||
        (dynamicDims != "" && dynamicShape != "")) {
        printf("%s - E - [XACL]: Can't set dynamicHWSize, dynamicBatch, dynamicDims or dynamicShape at the same time\n",
               GetTime().c_str());
        return FAILED;
    }

    // 整体执行step数
    uint32_t stepNumber = 0;
    // 整体计算耗时
    double totalSamplesTime = 0;
    // Context列表
    vector <aclrtContext> v_context;
    // 输入文件列表
    vector <string> v_inputFiles;

    Split(inputFiles, v_inputFiles, ",");

    // 判断是否存在-i入参，如不存在，则创建全零输入
    if (v_inputFiles.empty() != 1) {
        // 判断-i入参是否为目录
        bool b_isDir = false;
        struct stat s_stat;
        if (stat((char *) v_inputFiles[0].c_str(), &s_stat) == 0) {
            if (s_stat.st_mode & S_IFDIR) b_isDir = true;
        }

        if (b_isDir) {
            // 输入是目录时，遍历目录下所有输入文件
            printf("%s - I - [XACL]: Input type is director\n",
                   GetTime().c_str());
            vector <vector<string>> v_allInputFiles;
            for (uint32_t fileIdx = 0; fileIdx < v_inputFiles.size(); ++fileIdx) {
                vector <string> v_fileName;
                ScanFiles(v_fileName, v_inputFiles[fileIdx]);
                sort(v_fileName.begin(), v_fileName.end());
                v_allInputFiles.push_back(v_fileName);
            }

            // ACL接口初始化，包含aclInit，aclrtSetDevice，aclrtCreateContext等接口
            ret = InitResource(nodeId, v_context, dumpJson);
            if (ret != SUCCESS) {
                printf("%s - E - [XACL]: Init acl resource failed\n",
                       GetTime().c_str());
                return FAILED;
            } else {
                printf("%s - I - [XACL]: Init acl resource success\n",
                       GetTime().c_str());
            }

            // 加载om模型文件，包含aclmdlLoadFromFile等接口
            ret = LoadModelFromFile(modelFile, &modelId);
            if (ret != SUCCESS) {
                printf("%s - E - [XACL]: Load acl model from file failed\n",
                       GetTime().c_str());
                return FAILED;
            } else {
                printf("%s - I - [XACL]: Load acl model from file success\n",
                       GetTime().c_str());
            }

            // 创建模型描述，包含aclmdlCreateDesc，aclmdlGetDesc等接口
            aclmdlDesc *modelDesc = nullptr;
            modelDesc = CreateDesc(modelDesc, modelId);
            if (modelDesc == nullptr) {
                printf("%s - E - [XACL]: Create model description failed\n",
                       GetTime().c_str());
                // 创建模型描述失败时，释放模型，模型描述和模型上下文，包含ModelUnloadAndDescDestroy，DeviceContextDestroy等接口
                ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
                return FAILED;
            } else {
                printf("%s - I - [XACL]: Create model description success\n",
                       GetTime().c_str());
            }

            // 当输入是目录时，判断是否需要拼接输入
            if (mergeInput == "true") {
                // 当输入是目录时，且按Batch拼接输入标志为true，先拼接文件存放到拼接后目录
                // 创建拼接后文件目录，命名规则为：原文件目录 + 'batch' + N
                vector <string> v_mergedInputPath;
                for (uint32_t inputIdx = 0; inputIdx < v_inputFiles.size(); ++inputIdx) {
                    string inputIdxPath = v_inputFiles[inputIdx] + "_" + to_string(batchSize);
                    v_mergedInputPath.push_back(inputIdxPath);
                    ret = CreateDir(inputIdxPath);
                    if (ret != SUCCESS) {
                        return FAILED;
                    }
                }

                vector <vector<string>> v_allMergedInputFiles;
                // 循环每一个输入
                for (uint32_t inputIdx = 0; inputIdx < v_allInputFiles.size(); ++inputIdx) {
                    vector <string> v_inputFileName;
                    Split(v_inputFiles[inputIdx], v_inputFileName, "/");
                    string inputName = v_inputFileName[v_inputFileName.size() - 1];

                    printf("%s - I - [XACL]: Start to merge input %s\n",
                           GetTime().c_str(), inputName.c_str());
                    uint32_t fileIdx = 0, sliceNum = 0;
                    vector <string> v_mergedInputFiles;
                    for (; fileIdx < v_allInputFiles[0].size(); fileIdx += batchSize, ++sliceNum) {
                        // 若剩余文件不足一个batch则丢弃
                        if ((sliceNum + 1) * batchSize > v_allInputFiles[0].size()) {
                            break;
                        }
                        vector <string> v_inputFile;
                        string mergedNum = FormatInt(to_string(sliceNum), 5);
                        string mergedFile = v_mergedInputPath[inputIdx] + "/" + inputName + "_" + mergedNum + ".bin";
                        v_mergedInputFiles.push_back(mergedFile);
                        for (uint32_t b = 0; b < batchSize; ++b) {
                            v_inputFile.push_back(v_allInputFiles[inputIdx][fileIdx + b]);
                        }
                        MergeInputFile(v_inputFile, mergedFile);
                    }
                    v_allMergedInputFiles.push_back(v_mergedInputFiles);
                    printf("%s - I - [XACL]: Merge input %s to %s finished\n",
                           GetTime().c_str(), inputName.c_str(), v_mergedInputPath[inputIdx].c_str());

                }
                // 将拼接后的输入目录替换原始输入目录
                v_allInputFiles = v_allMergedInputFiles;
            }

            // 循环执行输入目录下所有文件
            for (uint32_t fileIdx = 0; fileIdx < v_allInputFiles[0].size(); ++fileIdx) {
                vector <string> v_singleInputFiles;
                for (uint32_t inputIdx = 0; inputIdx < v_allInputFiles.size(); ++inputIdx) {
                    printf("%s - I - [XACL]: The input file: %s is checked\n",
                           GetTime().c_str(), v_allInputFiles[inputIdx][fileIdx].c_str());
                    v_singleInputFiles.push_back(v_allInputFiles[inputIdx][fileIdx]);
                }

                // 创建输入
                aclmdlDataset *input;
                input = CreateInput(modelDesc, input, v_singleInputFiles, remoteDevice);
                if (input == nullptr) {
                    printf("%s - E - [XACL]: Create input data failed\n",
                           GetTime().c_str());
                    ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
                    return FAILED;
                } else {
                    printf("%s - I - [XACL]: Create input data success\n",
                           GetTime().c_str());
                }

                // 动态shape适配
                if (dynamicShape != "") {
                    if (dynamicOutSize != "") {
                        aclTensorDesc *inputDesc[v_inputFiles.size()];
                        ret = CreateInputDesc(input, v_inputFiles, dynamicShape, inputDesc);
                        if (ret != SUCCESS) {
                            printf("%s - E - [XACL]: Create input description failed\n",
                                   GetTime().c_str());
                            ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
                            return FAILED;
                        } else {
                            printf("%s - I - [XACL]: Create input description success\n",
                                   GetTime().c_str());
                        }

                    } else {
                        printf("%s - E - [XACL]: When dynamicShape function is enable, must set dynamicOutSize\n",
                               GetTime().c_str());
                        ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
                        return FAILED;
                    }
                }

                // 创建输出
                aclmdlDataset *output;
                output = CreateOutput(modelDesc, output, input, dynamicOutSize);
                if (output == nullptr) {
                    printf("%s - E - [XACL]: Create output data failed\n",
                           GetTime().c_str());
                    ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
                    return FAILED;
                } else {
                    printf("%s - I - [XACL]: Create output data success\n",
                           GetTime().c_str());
                }

                // 执行推理
                string outputFilesIdx = outputPrefix + "_" + FormatInt(to_string(fileIdx), 5);

                ret = Execute(modelId, input, output, outputFilesIdx, nodeId, modelDesc, v_context, remoteDevice,
                              loopNum, dynamicHWSize, dynamicBatch, dynamicDims, batchSize, totalSamplesTime);

                stepNumber++;

                if (ret != SUCCESS) {
                    printf("%s - E - [XACL]: Execute acl model failed\n",
                           GetTime().c_str());
                    ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
                    return FAILED;
                }
            }

            ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);

        } else {
            // 输入是指定bin文件
            printf("%s - I - [XACL]: Input type is bin file\n",
                   GetTime().c_str());

            int fileNum = int(v_inputFiles.size());
            printf("%s - I - [XACL]: The number of input files is: %d\n",
                   GetTime().c_str(), fileNum);

            // 判断输入是否为空文件
            printf("%s - I - [XACL]: Check whether the input files are empty\n",
                   GetTime().c_str());
            int inputIdx = 0;
            int emptyIdx = 0;
            int emptyList[127];
            for (int i = 0; i < 127; i++) {
                emptyList[i] = -1;
            }
            for (auto s: v_inputFiles) {
                ifstream binFile(s.c_str(), ifstream::binary);
                binFile.seekg(0, binFile.end);
                uint32_t binFileBufferLen = binFile.tellg();
                if (binFileBufferLen == 0) {
                    printf("%s - I - [XACL]: The input file: %s is empty\n",
                           GetTime().c_str(), s.c_str());
                    emptyList[emptyIdx] = inputIdx;
                    emptyIdx++;
                } else {
                    printf("%s - I - [XACL]: The input file: %s is checked\n",
                           GetTime().c_str(), s.c_str());
                }
                inputIdx++;
            }
            for (int i = 126; i >= 0; i--) {
                if (emptyList[i] != -1) {
                    v_inputFiles.erase(v_inputFiles.begin() + emptyList[i]);
                }
            }
            fileNum = int(v_inputFiles.size());
            printf("%s - I - [XACL]: The number of checked input files is: %d\n",
                   GetTime().c_str(), fileNum);

            // ACL接口初始化，包含aclInit，aclrtSetDevice，aclrtCreateContext等接口
            ret = InitResource(nodeId, v_context, dumpJson);
            if (ret != SUCCESS) {
                printf("%s - E - [XACL]: Init acl resource failed\n",
                       GetTime().c_str());
                return FAILED;
            } else {
                printf("%s - I - [XACL]: Init acl resource success\n",
                       GetTime().c_str());
            }

            // 加载om模型文件，包含aclmdlLoadFromFile等接口
            ret = LoadModelFromFile(modelFile, &modelId);
            if (ret != SUCCESS) {
                printf("%s - E - [XACL]: Load acl model from file failed\n",
                       GetTime().c_str());
                return FAILED;
            } else {
                printf("%s - I - [XACL]: Load acl model from file success\n",
                       GetTime().c_str());
            }

            // 创建模型描述，包含aclmdlCreateDesc，aclmdlGetDesc等接口
            aclmdlDesc *modelDesc = nullptr;
            modelDesc = CreateDesc(modelDesc, modelId);
            if (modelDesc == nullptr) {
                printf("%s - E - [XACL]: Create model description failed\n",
                       GetTime().c_str());
                // 创建模型描述失败时，释放模型，模型描述和模型上下文，包含ModelUnloadAndDescDestroy，DeviceContextDestroy等接口
                ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
                return FAILED;
            } else {
                printf("%s - I - [XACL]: Create model description success\n",
                       GetTime().c_str());
            }

            // 创建输入
            aclmdlDataset *input;
            input = CreateInput(modelDesc, input, v_inputFiles, remoteDevice);
            if (input == nullptr) {
                printf("%s - E - [XACL]: Create input data failed\n",
                       GetTime().c_str());
                ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
                return FAILED;
            } else {
                printf("%s - I - [XACL]: Create input data success\n",
                       GetTime().c_str());
            }

            // 动态shape适配
            if (dynamicShape != "") {
                if (dynamicOutSize != "") {
                    aclTensorDesc *inputDesc[v_inputFiles.size()];
                    ret = CreateInputDesc(input, v_inputFiles, dynamicShape, inputDesc);
                    if (ret != SUCCESS) {
                        printf("%s - E - [XACL]: Create input description failed\n",
                               GetTime().c_str());
                        ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
                        return FAILED;
                    } else {
                        printf("%s - I - [XACL]: Create input description success\n",
                               GetTime().c_str());
                    }

                } else {
                    printf("%s - E - [XACL]: When dynamicShape function is enable, must set dynamicOutSize\n",
                           GetTime().c_str());
                    ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
                    return FAILED;
                }
            }

            // 创建输出
            aclmdlDataset *output;
            output = CreateOutput(modelDesc, output, input, dynamicOutSize);
            if (output == nullptr) {
                printf("%s - E - [XACL]: Create output data failed\n",
                       GetTime().c_str());
                ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
                return FAILED;
            } else {
                printf("%s - I - [XACL]: Create output data success\n",
                       GetTime().c_str());
            }

            // 执行推理
            ret = Execute(modelId, input, output, outputPrefix, nodeId, modelDesc, v_context, remoteDevice,
                          loopNum, dynamicHWSize, dynamicBatch, dynamicDims, batchSize, totalSamplesTime);

            stepNumber++;

            if (ret != SUCCESS) {
                printf("%s - E - [XACL]: Execute acl model failed\n",
                       GetTime().c_str());
                ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
                return FAILED;
            }

            ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);

        }
    } else {
        // 不指定输入，创建全零输入
        printf("%s - I - [XACL]: Input type is empty, create all zero inputs\n",
               GetTime().c_str());

        // ACL接口初始化，包含aclInit，aclrtSetDevice，aclrtCreateContext等接口
        ret = InitResource(nodeId, v_context, dumpJson);
        if (ret != SUCCESS) {
            printf("%s - E - [XACL]: Init acl resource failed\n",
                   GetTime().c_str());
            return FAILED;
        } else {
            printf("%s - I - [XACL]: Init acl resource success\n",
                   GetTime().c_str());
        }

        // 加载om模型文件，包含aclmdlLoadFromFile等接口
        ret = LoadModelFromFile(modelFile, &modelId);
        if (ret != SUCCESS) {
            printf("%s - E - [XACL]: Load acl model from file failed\n",
                   GetTime().c_str());
            return FAILED;
        } else {
            printf("%s - I - [XACL]: Load acl model from file success\n",
                   GetTime().c_str());
        }

        // 创建模型描述，包含aclmdlCreateDesc，aclmdlGetDesc等接口
        aclmdlDesc *modelDesc = nullptr;
        modelDesc = CreateDesc(modelDesc, modelId);
        if (modelDesc == nullptr) {
            printf("%s - E - [XACL]: Create model description failed\n",
                   GetTime().c_str());
            // 创建模型描述失败时，释放模型，模型描述和模型上下文，包含ModelUnloadAndDescDestroy，DeviceContextDestroy等接口
            ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
            return FAILED;
        } else {
            printf("%s - I - [XACL]: Create model description success\n",
                   GetTime().c_str());
        }

        // 创建输入
        aclmdlDataset *input;
        input = CreateZeroInput(modelDesc, input, v_inputFiles, remoteDevice);
        if (input == nullptr) {
            printf("%s - E - [XACL]: Create input data failed\n",
                   GetTime().c_str());
            ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
            return FAILED;
        } else {
            printf("%s - I - [XACL]: Create input data success\n",
                   GetTime().c_str());
        }

        // 动态shape适配
        if (dynamicShape != "") {
            if (dynamicOutSize != "") {
                aclTensorDesc *inputDesc[v_inputFiles.size()];
                ret = CreateInputDesc(input, v_inputFiles, dynamicShape, inputDesc);
                if (ret != SUCCESS) {
                    printf("%s - E - [XACL]: Create input description failed\n",
                           GetTime().c_str());
                    ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
                    return FAILED;
                } else {
                    printf("%s - I - [XACL]: Create input description success\n",
                           GetTime().c_str());
                }
            } else {
                printf("%s - E - [XACL]: When dynamicShape function is enable, must set dynamicOutSize\n",
                       GetTime().c_str());
                ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
                return FAILED;
            }
        }

        // 创建输出
        aclmdlDataset *output;
        output = CreateOutput(modelDesc, output, input, dynamicOutSize);
        if (output == nullptr) {
            printf("%s - E - [XACL]: Create output data failed\n",
                   GetTime().c_str());
            ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
            return FAILED;
        } else {
            printf("%s - I - [XACL]: Create output data success\n",
                   GetTime().c_str());
        }

        // 执行推理
        ret = Execute(modelId, input, output, outputPrefix, nodeId, modelDesc, v_context, remoteDevice,
                      loopNum, dynamicHWSize, dynamicBatch, dynamicDims, batchSize, totalSamplesTime);

        stepNumber++;

        if (ret != SUCCESS) {
            printf("%s - E - [XACL]: Execute acl model failed\n",
                   GetTime().c_str());
            ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);
            return FAILED;
        }

        ReleaseAllModelResource(modelId, nodeId, modelDesc, v_context);

    }

    uint32_t samplesNumber = stepNumber * batchSize;
    double totalAverageTime = totalSamplesTime / stepNumber;
    double totalAverageFPS = 1000 * samplesNumber / totalSamplesTime;

    printf("%s - I - [XACL]: %d samples average NPU inference time of %d steps: %f ms %4.2f fps\n",
           GetTime().c_str(), samplesNumber, stepNumber, totalAverageTime, totalAverageFPS);

    ret = WriteResult(modelFile, outputPath, samplesNumber, stepNumber, totalAverageTime, totalAverageFPS);

    if (ret != SUCCESS) {
        printf("%s - E - [XACL]: Write acl result failed\n",
               GetTime().c_str());
        return FAILED;
    }

    return SUCCESS;
}