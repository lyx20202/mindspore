/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef NNACL_FP32_GRAD_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_
#define NNACL_FP32_GRAD_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void ForwardPostExecute(const float *labels, const float *logits, float *grads, float *output2,
                        size_t number_of_classes, int batch_size);

#ifdef __cplusplus
}
#endif

#endif  // NNACL_FP32_GRAD_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_
