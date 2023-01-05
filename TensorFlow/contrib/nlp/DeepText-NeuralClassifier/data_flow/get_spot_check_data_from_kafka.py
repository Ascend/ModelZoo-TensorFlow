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
# ==============================================================================

import kafka_handler
import time

# kafka.
kafka_user = ""
kafka_password = ""
kakfa_servers = ""
kafka_topic = ""
kafka_group = ""
kafka_result_consumer = kafka_handler.Consumer(kakfa_servers, kafka_user, kafka_password,
                                               kafka_topic, kafka_group)


def main():
    now_time = time.strftime("%Y-%m-%d", time.localtime())
    f = open("data_kafka_{}".format(now_time), "w+")
    for message in kafka_result_consumer.feed():
        try:
            f.write(message.decode("utf-8") + '\n')
        except Exception as e:
            print(e)
    f.close()


if __name__ == '__main__':
    main()

