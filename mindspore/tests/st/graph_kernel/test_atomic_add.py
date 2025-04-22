# Copyright 2020 Huawei Technologies Co., Ltd
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

import numpy as np
from tests.mark_utils import arg_mark
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations as P


class SumOutNet(Cell):
    def __init__(self):
        super(SumOutNet, self).__init__()
        self.square = P.Square()
        self.sum = P.ReduceSum()

    def construct(self, x):
        mul_res = self.square(x)
        return self.sum(mul_res, (0,))


class SingleOutNet(Cell):
    def __init__(self):
        super(SingleOutNet, self).__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.sum = P.ReduceSum()

    def construct(self, x, y):
        mul_res = self.mul(x, y)
        sum_res = self.sum(mul_res, ())
        return self.add(sum_res, x)


class MultiOutNet(Cell):
    def __init__(self):
        super(MultiOutNet, self).__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.sum = P.ReduceSum()

    def construct(self, x, y):
        add_res = self.add(x, y)
        mul_res = self.mul(add_res, add_res)
        sum_res = self.sum(mul_res, ())
        return self.add(add_res, sum_res)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_atomic_add_sum_output():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
    np.random.seed(0)
    input_x = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)

    expect = np.sum(np.square(input_x), axis=(0,))

    net = SumOutNet()
    result = net(Tensor(input_x))

    res = np.allclose(expect, result.asnumpy(), rtol=1.e-4, atol=1.e-7, equal_nan=True)
    assert res


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_atomic_add_single_output():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
    np.random.seed(0)
    input_x = np.random.normal(0, 1, [2, 2, 2, 256]).astype(np.float32)
    input_y = np.random.normal(0, 1, [2, 2, 2, 256]).astype(np.float32)

    expect = np.sum(input_x * input_y) + input_x

    net = SingleOutNet()
    result = net(Tensor(input_x), Tensor(input_y))

    res = np.allclose(expect, result.asnumpy(), rtol=1.e-4, atol=1.e-7, equal_nan=True)
    assert res


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_atomic_add_multi_output():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
    np.random.seed(0)
    input_x = np.random.normal(0, 1, [2, 2, 2, 256]).astype(np.float32)
    input_y = np.random.normal(0, 1, [2, 2, 2, 256]).astype(np.float32)

    expect = np.sum(np.square(input_x + input_y)) + (input_x + input_y)

    net = MultiOutNet()
    result = net(Tensor(input_x), Tensor(input_y))

    res = np.allclose(expect, result.asnumpy(), rtol=1.e-4, atol=1.e-7, equal_nan=True)
    assert res
