# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore
from mindspore import ops, nn, context, Tensor
from .test_grad_of_dynamic import TestDynamicGrad


class TestAddV2(nn.Cell):
    def __init__(self):
        super(TestAddV2, self).__init__()
        self.ops = ops.operations.math_ops.AddV2()

    def construct(self, x, y):
        return self.ops(x, y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_mul_dynamic_shape():
    """
    Feature: AddV2 Grad DynamicShape.
    Description: Test case of dynamic shape for AddV2 grad operator on CPU.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_dynamic = TestDynamicGrad(TestAddV2())
    input_x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
    input_y = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
    x = [input_x, input_y]
    test_dynamic.test_dynamic_grad_net(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_mul_dynamic_rank():
    """
    Feature: AddV2 Grad DynamicShape.
    Description: Test case of dynamic rank for AddV2 grad operator on CPU.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_dynamic = TestDynamicGrad(TestAddV2())
    input_x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
    input_y = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
    x = [input_x, input_y]
    test_dynamic.test_dynamic_grad_net(x, True)
