# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
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

import sys
import locale

system_locale = locale.getdefaultlocale()[0]
# system_locale may be nil
system_language = system_locale[0:2] if system_locale is not None else "en"
if system_language not in ['en','ru','zh']:
    system_language = 'en'

windows_font_name_map = {
    'en' : 'cour',
    'ru' : 'cour',
    'zh' : 'simsun_01'
}

darwin_font_name_map = {
    'en' : 'cour',
    'ru' : 'cour',
    'zh' : 'Apple LiSung Light'
}

linux_font_name_map = {
    'en' : 'cour',
    'ru' : 'cour',
    'zh' : 'cour'
}

def get_default_ttf_font_name():
    platform = sys.platform
    if platform[0:3] == 'win': return windows_font_name_map.get(system_language, 'cour')
    elif platform == 'darwin': return darwin_font_name_map.get(system_language, 'cour')
    else: return linux_font_name_map.get(system_language, 'cour')

SID_HOT_KEY = 1

if system_language == 'en':
    StringsDB = {'S_HOT_KEY' : 'hot key'}
elif system_language == 'ru':
    StringsDB = {'S_HOT_KEY' : 'горячая клавиша'}    
elif system_language == 'zh':
    StringsDB = {'S_HOT_KEY' : '热键'}   
    