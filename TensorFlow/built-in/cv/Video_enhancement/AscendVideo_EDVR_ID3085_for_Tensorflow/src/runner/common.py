# Copyright 2022 Huawei Technologies Co., Ltd
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
from collections import OrderedDict

from src.utils.klass import Singleton
from src.utils.logger import logger


class _NameSpace(metaclass=Singleton):
    """A common name-space class to record, sort and retrieve the tensorflow ops.

    Attributes:
        GeneratorLoss: the scope of the loss ops of the generator.
        DiscriminatorLoss: the scope of the loss ops of the discriminator.
        GeneratorVarScope: the scope of the variables in the generator.
        DiscriminatorVarScope: the scope of the variables in the discriminator.
        PerceptualVarScope: the scope of the variables in the perceptual module.
        Summary: the scope of the summary ops.
        GeneratorRunOp: the scope of the running ops, i.e. train_op, lr_update_op, 
            of the generator.
        DiscriminatorRunOp: the scope of the running ops, i.e. train_op, lr_update_op, 
            of the discriminator.
        InputField: the scope of the input tensor and ops.
        OutputField: the scope of the output tensor and ops.

    Example: record the losses in the GeneratorLoss scope, retrieve and add them to get
        the final total loss for training.

        >>> from src.runner.common import name_space
        >>> l1_loss = compute_loss1(pred, gt)
        >>> name_space.add_to_collection(name_space.GeneratorLoss, 'l1_loss', l1_loss)
        >>> l2_loss = compute_loss2(pred, gt)
        >>> name_space.add_to_collection(name_space.GeneratorLoss, 'l2_loss', l2_loss)
        >>> ...
        >>> losses_dict = name_space.get_collection(name_space.GeneratorLoss)
        >>> total_loss = tf.add_n(list(losses_dict.values()))   # l1_loss + l2_loss
    """
    __scopes = dict(
        GeneratorLoss='gen_loss',
        DiscriminatorLoss='dis_loss',
        GeneratorVarScope='gen_var',
        DiscriminatorVarScope='dis_var',
        PerceptualVarScope='percep_var',
        Summary='summary',
        GeneratorRunOp='gen_op',
        DiscriminatorRunOp='dis_op',
        InputField='input',
        OutputField='output',
    )

    __collections = dict()

    def __init__(self):
        for key, value in self.__scopes.items():
            setattr(self, key, value)
            self.__collections[value] = OrderedDict()

    def add_to_collection(self, namespace, key, value):
        assert namespace in self.__scopes.values()
        if key in self.__collections[namespace]:
            logger.warn(f'Key "{key}" has already exists in scope "{namespace}".')
        self.__collections[namespace][key] = value

    def add_to_collections(self, namespaces, key, value):
        assert isinstance(namespaces, (list, tuple))
        for name in namespaces:
            self.add_to_collection(name, key, value)

    def get_collection(self, namespace):
        return self.__collections[namespace]

    def get_op(self, namespace, opname):
        return self.__collections[namespace][opname]


name_space = _NameSpace()
