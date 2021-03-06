/* *
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* */

#include "post_process.h"
#include "util.h"
#include <vector>
#include <cstring>
#include <memory>
#include <fstream>
#include "stdio.h"
#include <sys/time.h>
#include <unistd.h>
#include <time.h> 
#include <dirent.h> 
#include <stdarg.h>
#include <iostream>

#include <unistd.h>
#include <thread>
#include <algorithm>
#include <libgen.h>
#include <string.h>
#include <getopt.h>
#include <map>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sstream>
#include <cerrno> 
#include <errno.h> 
#include <sys/errno.h>
#include <unordered_map>


extern int processedCnt;

extern Config cfg;
extern DataFrame outputDataframe;
extern aclError ret;
int topNum = 5;

extern int processedCnt;

aclError SaveBinPostprocess()
{
    aclError retVal;

    LOG("save batch %d start\n", processedCnt);
    DataFrame dataframe = outputDataframe;
    std::vector<std::string>& inferFile_vec = outputDataframe.fileNames;
    aclmdlDataset* output = dataframe.dataset;

    std::string resultFolder = cfg.outDir + "/" + cfg.modelType;
	LOG("resultFolder is %s.\n",resultFolder.c_str());
    DIR* op = opendir(resultFolder.c_str());
    if (NULL == op){
        mkdir(resultFolder.c_str(), 00775);
    }else{
        closedir(op);
    }
	
	

    for (size_t i = 0; i < cfg.outputNum; ++i)
    {        
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
		uint32_t length = aclGetDataBufferSize(dataBuffer);
        uint32_t len;
		void* outHostData = NULL;
		uint32_t size = 0;
        len = cfg.outputInfo[i].size;
		LOG("====SaveBinPostprocess.dataset output,outputData:%p,size:%p\n", output, data);        
        LOG("cfg.outputNum is %d, output[%d] real data len %d %d\n",cfg.outputNum, i, len,length);

        ret = aclrtMallocHost(&outHostData, len);
        if (ret != ACL_ERROR_NONE) {
            LOG("Malloc host failed.\n");
            return 1;
        }

        ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_ERROR_NONE) {
            LOG("Copy device to host failed.\n");
            aclrtFreeHost(outHostData);
            return 1;
        }

        uint32_t eachSize = len / cfg.batchSize;
        //LOG("inferenceIndex=%d, out_index=%d, len=%d, eachSize=%d\n", inferenceIndex, i, len,eachSize);

			for (size_t j = 0; j < inferFile_vec.size(); j++)
	        {
	            FILE* outputFile;
	            std::string framename = inferFile_vec[j];
	            std::size_t dex = (framename).find_first_of(".");
	            std::string inputFileName = (framename).erase(dex);

	            
	            if (cfg.modelType.compare(0, 6, "resnet") == 0){
	                outputFile = fopen((resultFolder + "/" + "davinci_" + inputFileName + "_"  + "output" + ".bin").c_str(), "wb");
	            }else{
	                outputFile = fopen((resultFolder + "/" + "davinci_" + inputFileName + "_"  + "output" + std::to_string(i) + ".bin").c_str(), "wb");
	            }
	            
	            if (NULL == outputFile){
	                aclrtFreeHost(outHostData);
	                return 1;
	            }

	            fwrite((uint8_t *)outHostData + size, eachSize, sizeof(char), outputFile);
	            fclose(outputFile);
				size = size + eachSize;
	        }

        
        
        ret = aclrtFreeHost(outHostData);
        if (ret != ACL_ERROR_NONE) {
            LOG("Free output host failed.\n");
        }
    }
    
    (void)DestroyDatasetResurce(outputDataframe.dataset, 0);
    
    LOG("save batch %d done\n", processedCnt);
    return ACL_ERROR_NONE;
}
