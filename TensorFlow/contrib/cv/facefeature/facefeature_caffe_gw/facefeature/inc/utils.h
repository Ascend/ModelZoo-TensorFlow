/**
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

* File utils.h
* Description: handle file operations
*/
#pragma once
#include <iostream>
#include <vector>
#include "acl/acl.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "opencv2/imgproc/types_c.h"

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

#define RGBU8_IMAGE_SIZE(width, height) ((width) * (height) * 3)
#define YUV420SP_SIZE(width, height) ((width) * (height) * 3 / 2)

#define ALIGN_UP(num, align) (((num) + (align) - 1) & ~((align) - 1))
#define ALIGN_UP2(num) ALIGN_UP(num, 2)
#define ALIGN_UP16(num) ALIGN_UP(num, 16)
#define ALIGN_UP128(num) ALIGN_UP(num, 128)

#define CHECK_ODD(NUM) ((((NUM) % (2)) != (0)) ? (NUM) : ((NUM) - (1)))
#define CHECK_EVEN(NUM) ((((NUM) % (2)) == (0)) ? (NUM) : ((NUM) - (1)))

#define SHARED_PRT_DVPP_BUF(buf) (shared_ptr<uint8_t>((uint8_t *)(buf), [](uint8_t* p) { acldvppFree(p); }))
#define SHARED_PRT_U8_BUF(buf) (shared_ptr<uint8_t>((uint8_t *)(buf), [](uint8_t* p) { delete[](p); }))


template<class Type>
std::shared_ptr<Type> MakeSharedNoThrow() {
    try {
        return std::make_shared<Type>();
    }
    catch (...) {
        return nullptr;
    }
}

#define MAKE_SHARED_NO_THROW(memory, memory_type) \
    do { \
            memory = MakeSharedNoThrow<memory_type>(); \
    }while(0);    

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
}Result;

struct Resolution {
    uint32_t width = 0;
    uint32_t height = 0;
};

struct ImageData {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t alignWidth = 0;
    uint32_t alignHeight = 0;
    uint32_t size = 0;
    std::shared_ptr<uint8_t> data;
};

struct Rect {
    uint32_t ltX = 0;
    uint32_t ltY = 0;
    uint32_t rbX = 0;
    uint32_t rbY = 0;
};

struct BBox {
    Rect rect;
    uint32_t score;
    std::string text;
};

struct FaceInfo {
    uint32_t featureSize;
    char fileName[50];
    float feature[512];
};

/**
 * Utils
 */
class Utils {
public:

    /**
    * @brief create device buffer of pic
    * @param [in] picDesc: pic desc
    * @param [in] PicBufferSize: aligned pic size
    * @return device buffer of pic
    */
    static bool is_directory(const std::string &path);

    static bool is_path_exist(const std::string &path);

    static void split_path(const std::string &path, std::vector<std::string> &path_vec);

    static void get_all_files(const std::string &path, std::vector<std::string> &file_vec);

    static void get_path_files(const std::string &path, std::vector<std::string> &file_vec);
    static void* copy_data_to_device(void* data, uint32_t dataSize, aclrtMemcpyKind policy);
    static void* copy_data_device_to_local(void* deviceData, uint32_t dataSize);
    static void* copy_data_host_to_device(void* deviceData, uint32_t dataSize);
    static void* copy_data_device_to_device(void* deviceData, uint32_t dataSize);
    static Result copy_image_data_to_device(ImageData& imageDevice, ImageData srcImage, aclrtRunMode mode);
};

