#
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
#
from npu_bridge.npu_init import *
from contextlib import ExitStack

from zmq.decorators import _Decorator

__all__ = ['multi_socket']

from functools import wraps

import zmq


class _MyDecorator(_Decorator):
    def __call__(self, *dec_args, **dec_kwargs):
        kw_name, dec_args, dec_kwargs = self.process_decorator_args(*dec_args, **dec_kwargs)
        num_socket_str = dec_kwargs.pop('num_socket')

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                num_socket = getattr(args[0], num_socket_str)
                targets = [self.get_target(*args, **kwargs) for _ in range(num_socket)]
                with ExitStack() as stack:
                    for target in targets:
                        obj = stack.enter_context(target(*dec_args, **dec_kwargs))
                        args = args + (obj,)

                    return func(*args, **kwargs)

            return wrapper

        return decorator


class _SocketDecorator(_MyDecorator):
    def process_decorator_args(self, *args, **kwargs):
        """Also grab context_name out of kwargs"""
        kw_name, args, kwargs = super(_SocketDecorator, self).process_decorator_args(*args, **kwargs)
        self.context_name = kwargs.pop('context_name', 'context')
        return kw_name, args, kwargs

    def get_target(self, *args, **kwargs):
        """Get context, based on call-time args"""
        context = self._get_context(*args, **kwargs)
        return context.socket

    def _get_context(self, *args, **kwargs):
        if self.context_name in kwargs:
            ctx = kwargs[self.context_name]

            if isinstance(ctx, zmq.Context):
                return ctx

        for arg in args:
            if isinstance(arg, zmq.Context):
                return arg
        # not specified by any decorator
        return zmq.Context.instance()


def multi_socket(*args, **kwargs):
    return _SocketDecorator()(*args, **kwargs)

