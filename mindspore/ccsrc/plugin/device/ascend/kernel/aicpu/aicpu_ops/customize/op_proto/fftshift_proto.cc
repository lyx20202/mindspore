/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "custom_op_proto/cust_math_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
IMPLEMT_COMMON_INFERFUNC(FFTShiftInferShape) {
  TensorDesc out_desc = op.GetOutputDescByName("input");
  out_desc.SetDataType(op.GetInputDescByName("input").GetDataType());
  out_desc.SetShape(op.GetInputDescByName("input").GetShape());
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to update output desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(FFTShift, FFTShiftInferShape);
CUST_COMMON_INFER_FUNC_REG(IFFTShift, FFTShiftInferShape);
}  // namespace ge