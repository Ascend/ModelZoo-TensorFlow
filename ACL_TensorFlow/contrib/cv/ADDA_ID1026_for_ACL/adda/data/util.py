"""
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
"""
import logging
import os.path

import requests

logger = logging.getLogger(__name__)


def maybe_download(url, dest):
    """Download the url to dest if necessary, optionally checking file
    integrity.
    """
    if not os.path.exists(dest):
        logger.info('Downloading %s to %s', url, dest)
        download(url, dest)


def download(url, dest):
    """Download the url to dest, overwriting dest if it already exists."""
    response = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

