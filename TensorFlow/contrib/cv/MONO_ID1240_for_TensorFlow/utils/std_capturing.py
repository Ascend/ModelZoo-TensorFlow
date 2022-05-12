#!/usr/bin/env python
# coding=utf-8

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
# ============================================================================

from __future__ import division, print_function, unicode_literals
from npu_bridge.npu_init import *
import os
import sys
import subprocess
from threading import Timer
from contextlib import contextmanager

def flush():
    """Try to flush all stdio buffers, both from python and from C."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (AttributeError, ValueError, IOError):
        pass  # unsupported


@contextmanager
def capture_outputs(filename):
    """Duplicate stdout and stderr to a file on the file descriptor level."""
    # with NamedTemporaryFile(mode='w+') as target:
    with open(filename, 'a+') as target:
        original_stdout_fd = 1
        original_stderr_fd = 2
        target_fd = target.fileno()

        # Save a copy of the original stdout and stderr file descriptors
        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)

        tee_stdout = subprocess.Popen(
            ['tee', '-a', '/dev/stderr'], start_new_session=True,
            stdin=subprocess.PIPE, stderr=target_fd, stdout=1)
        tee_stderr = subprocess.Popen(
            ['tee', '-a', '/dev/stderr'], start_new_session=True,
            stdin=subprocess.PIPE, stderr=target_fd, stdout=2)

        flush()
        os.dup2(tee_stdout.stdin.fileno(), original_stdout_fd)
        os.dup2(tee_stderr.stdin.fileno(), original_stderr_fd)

        try:
            yield
        finally:
            flush()

            # then redirect stdout back to the saved fd
            tee_stdout.stdin.close()
            tee_stderr.stdin.close()

            # restore original fds
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.dup2(saved_stderr_fd, original_stderr_fd)

            # wait for completion of the tee processes with timeout
            # implemented using a timer because timeout support is py3 only
            def kill_tees():
                tee_stdout.kill()
                tee_stderr.kill()

            tee_timer = Timer(1, kill_tees)
            try:
                tee_timer.start()
                tee_stdout.wait()
                tee_stderr.wait()
            finally:
                tee_timer.cancel()

            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
