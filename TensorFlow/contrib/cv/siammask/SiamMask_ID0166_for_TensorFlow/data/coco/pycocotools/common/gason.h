//# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//#
//# Licensed under the Apache License, Version 2.0 (the "License");
//# you may not use this file except in compliance with the License.
//# You may obtain a copy of the License at
//#
//#     http://www.apache.org/licenses/LICENSE-2.0
//#
//# Unless required by applicable law or agreed to in writing, software
//# distributed under the License is distributed on an "AS IS" BASIS,
//# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//# See the License for the specific language governing permissions and
//# limitations under the License.
//# ============================================================================
//# Licensed under The MIT License
//# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
//# limitations under the License.
//# ============================================================================
//# Copyright 2020 Huawei Technologies Co., Ltd
//#
//# Licensed under the Apache License, Version 2.0 (the "License");
//# you may not use this file except in compliance with the License.
//# You may obtain a copy of the License at
//#
//#     http://www.apache.org/licenses/LICENSE-2.0
//#
//# Unless required by applicable law or agreed to in writing, software
//# distributed under the License is distributed on an "AS IS" BASIS,
//# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//# See the License for the specific language governing permissions and
//# limitations under the License.

// https://github.com/vivkin/gason - pulled January 10, 2016
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

enum JsonTag {
    JSON_NUMBER = 0,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT,
    JSON_TRUE,
    JSON_FALSE,
    JSON_NULL = 0xF
};

struct JsonNode;

#define JSON_VALUE_PAYLOAD_MASK 0x00007FFFFFFFFFFFULL
#define JSON_VALUE_NAN_MASK 0x7FF8000000000000ULL
#define JSON_VALUE_TAG_MASK 0xF
#define JSON_VALUE_TAG_SHIFT 47

union JsonValue {
    uint64_t ival;
    double fval;

    JsonValue(double x)
        : fval(x) {
    }
    JsonValue(JsonTag tag = JSON_NULL, void *payload = nullptr) {
        assert((uintptr_t)payload <= JSON_VALUE_PAYLOAD_MASK);
        ival = JSON_VALUE_NAN_MASK | ((uint64_t)tag << JSON_VALUE_TAG_SHIFT) | (uintptr_t)payload;
    }
    bool isDouble() const {
        return (int64_t)ival <= (int64_t)JSON_VALUE_NAN_MASK;
    }
    JsonTag getTag() const {
        return isDouble() ? JSON_NUMBER : JsonTag((ival >> JSON_VALUE_TAG_SHIFT) & JSON_VALUE_TAG_MASK);
    }
    uint64_t getPayload() const {
        assert(!isDouble());
        return ival & JSON_VALUE_PAYLOAD_MASK;
    }
    double toNumber() const {
        assert(getTag() == JSON_NUMBER);
        return fval;
    }
    char *toString() const {
        assert(getTag() == JSON_STRING);
        return (char *)getPayload();
    }
    JsonNode *toNode() const {
        assert(getTag() == JSON_ARRAY || getTag() == JSON_OBJECT);
        return (JsonNode *)getPayload();
    }
};

struct JsonNode {
    JsonValue value;
    JsonNode *next;
    char *key;
};

struct JsonIterator {
    JsonNode *p;

    void operator++() {
        p = p->next;
    }
    bool operator!=(const JsonIterator &x) const {
        return p != x.p;
    }
    JsonNode *operator*() const {
        return p;
    }
    JsonNode *operator->() const {
        return p;
    }
};

inline JsonIterator begin(JsonValue o) {
    return JsonIterator{o.toNode()};
}
inline JsonIterator end(JsonValue) {
    return JsonIterator{nullptr};
}

#define JSON_ERRNO_MAP(XX)                           \
    XX(OK, "ok")                                     \
    XX(BAD_NUMBER, "bad number")                     \
    XX(BAD_STRING, "bad string")                     \
    XX(BAD_IDENTIFIER, "bad identifier")             \
    XX(STACK_OVERFLOW, "stack overflow")             \
    XX(STACK_UNDERFLOW, "stack underflow")           \
    XX(MISMATCH_BRACKET, "mismatch bracket")         \
    XX(UNEXPECTED_CHARACTER, "unexpected character") \
    XX(UNQUOTED_KEY, "unquoted key")                 \
    XX(BREAKING_BAD, "breaking bad")                 \
    XX(ALLOCATION_FAILURE, "allocation failure")

enum JsonErrno {
#define XX(no, str) JSON_##no,
    JSON_ERRNO_MAP(XX)
#undef XX
};

const char *jsonStrError(int err);

class JsonAllocator {
    struct Zone {
        Zone *next;
        size_t used;
    } *head = nullptr;

public:
    JsonAllocator() = default;
    JsonAllocator(const JsonAllocator &) = delete;
    JsonAllocator &operator=(const JsonAllocator &) = delete;
    JsonAllocator(JsonAllocator &&x) : head(x.head) {
        x.head = nullptr;
    }
    JsonAllocator &operator=(JsonAllocator &&x) {
        head = x.head;
        x.head = nullptr;
        return *this;
    }
    ~JsonAllocator() {
        deallocate();
    }
    void *allocate(size_t size);
    void deallocate();
};

int jsonParse(char *str, char **endptr, JsonValue *value, JsonAllocator &allocator);
