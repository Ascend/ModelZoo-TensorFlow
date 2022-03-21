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

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class QXMainWindow(QWidget):
    """
    Custom mainwindow class that provides global single instance and event listeners
    """
    inst = None
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        if QXMainWindow.inst is not None:
            raise Exception("QXMainWindow can only be one.")        
        QXMainWindow.inst = self
        
        self.keyPressEvent_listeners = []
        self.keyReleaseEvent_listeners = []
        self.setFocusPolicy(Qt.WheelFocus)
        
    def add_keyPressEvent_listener(self, func):
        self.keyPressEvent_listeners.append (func)
        
    def add_keyReleaseEvent_listener(self, func):
        self.keyReleaseEvent_listeners.append (func)
        
    def keyPressEvent(self, ev):
        super().keyPressEvent(ev)        
        for func in self.keyPressEvent_listeners:
            func(ev)
            
    def keyReleaseEvent(self, ev):
        super().keyReleaseEvent(ev)        
        for func in self.keyReleaseEvent_listeners:
            func(ev)