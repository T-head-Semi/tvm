/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file get_aitrace_data.h
 * \brief get aitrace data for profiler.
 */
#ifndef TVM_RELAY_ANALYSIS_GET_AITRACE_DATA_H_
#define TVM_RELAY_ANALYSIS_GET_AITRACE_DATA_H_

#include <tvm/ir/error.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/vision.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/type.h>

#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../backend/utils.h"
#include "profiler_parser.h"

namespace tvm {
namespace relay {
namespace aitrace {

using AiTraceDataFrame = Map<String, Map<String, ObjectRef> >;
using FCalAmount = runtime::TypedPackedFunc<AiTraceDataFrame(const Call& call_node)>;
using FMemory = runtime::TypedPackedFunc<AiTraceDataFrame(const Call& call_node)>;

using FOpName = runtime::TypedPackedFunc<String()>;

const std::unordered_map<std::string, uint16_t> op_map = {
    {"add", RELAY_OP_ADD},
    {"nn.avg_pool2d", RELAY_OP_AVGPOOL_2D},
    {"nn.batch_flatten", RELAY_OP_BATCH_FLATTEN},
    {"nn.batch_norm", RELAY_OP_BATCH_NORM},
    {"nn.bias_add", RELAY_OP_BIAS_ADD},
    {"concatenate", RELAY_OP_CONCATENATE},
    {"nn.conv2d", RELAY_OP_CONV2D},
    {"nn.conv2d_transpose", RELAY_OP_CONV2D_TRANSPOSE},
    {"nn.dense", RELAY_OP_DENSE},
    {"nn.dropout", RELAY_OP_DROPOUT},
    {"nn.global_avg_pool2d", RELAY_OP_GLOBAL_AVGPOOL_2D},
    {"nn.global_max_pool2d", RELAY_OP_GLOBAL_MAXPOOL_2D},
    {"nn.lrn", RELAY_OP_LRN},
    {"nn.l2_normalize", RELAY_OP_L2NORMALIZE},
    {"maximum", RELAY_OP_MAXIMUM},
    {"nn.max_pool2d", RELAY_OP_MAXPOOL_2D},
    {"vision.max_pool2d_location", RELAY_OP_MAXPOOL_2D_LOCATION},
    {"max_pool2d_with_argmax", RELAY_OP_MAXPOOL_2D_WITH_ARGMAX},
    {"multiply", RELAY_OP_MULTIPLY},
    {"nn.prelu", RELAY_OP_PRELU},
    {"vision.proposal", RELAY_OP_PROPOSAL},
    {"vision.psroipooling", RELAY_OP_PSROIPOOLING},
    {"nn.relu", RELAY_OP_RELU},
    {"reshape", RELAY_OP_RESHAPE},
    {"vision.roi_pool", RELAY_OP_ROIPOOL},
    {"sigmoid", RELAY_OP_SIGMOID},
    {"nn.softmax", RELAY_OP_SOFTMAX},
    {"split", RELAY_OP_SPLIT},
    {"strided_slice", RELAY_OP_STRIDED_SLICE},
    {"tanh", RELAY_OP_TANH},
    {"transpose", RELAY_OP_TRANSPOSE},
    {"vision.unpooling", RELAY_OP_UNPOOLING},
    {"nn.upsampling", RELAY_OP_UNPSAMPLING}};

/*!
 * \brief Get values accordding to specific key in unordered_map, support for default value.
 * \param data The unordered_map to search.
 * \param key The key to search.
 * \param defualt_value return this default value if not find key in data.
 * \return result value.
 */
inline int64_t GetMapValues(const std::unordered_map<std::string, int64_t>& data,
                            const std::string& key, int64_t default_value) {
  int64_t result = default_value;
  if (data.find(key) != data.end()) {
    result = data.at(key);
  }
  return result;
}

/*! \brief Get cumulative multiplication in an array. */
inline int64_t GetCartesianProd(Array<IndexExpr> arr) {
  int64_t ret = 1;
  for (size_t i = 0; i < arr.size(); i++) {
    const auto* intImm = arr[i].as<IntImmNode>();
    ret *= static_cast<int64_t>(intImm->value);
  }
  return ret;
}

/*! \brief Replace sub-string with target string.
 *
 * \param str The string to be changed.
 * \param from The sub-string that will be replace.
 * \param to The target string.
 * \return True if replaced successfully.
 */
inline bool ReplaceStr(std::string* str, const std::string& from, const std::string& to) {
  size_t start_pos = str->find(from);
  if (start_pos == std::string::npos) {
    return false;
  }
  str->replace(start_pos, from.length(), to);
  return true;
}

/*! Replace some invalid chars from string. */
inline void ReplaceInvalidSymbol(std::string* str, const std::string& to) {
  std::vector<std::string> invalid_symbols = {".", "/", "\\", ":"};

  for (auto i : invalid_symbols) {
    ReplaceStr(str, i, to);
  }
}

/*! \brief Get cumulative multiplication in an array. */
inline int64_t GetSize(std::vector<int64_t> shape) {
  CHECK_GT(shape.size(), 0) << "The shape should'n be empty...";
  int64_t size = 1;
  for (auto s : shape) {
    size *= s;
  }
  return size;
}

/*! \brief Convert list of AiTraceDataFrame into ai trace data. */
AiTraceData Convert2ATData(Array<AiTraceDataFrame> origin_data);

/*! \brief Count calculation amount of Pool layers. */
std::unordered_map<std::string, int64_t> PoolCalAmountCommon(
    const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape,
    const std::vector<int64_t>& kernel_shape, const std::string& data_layout,
    const std::string& op_type, const bool& is_global);

/*! \brief Count calculation amount of Pool layers. */
template <typename T>
AiTraceDataFrame GetPoolCalAmountCommon(const Call& call_node, T attrs, const std::string& op_type,
                                        const bool& is_global);

/*! \brief Count calculation amount of relu layers. */
AiTraceDataFrame GetReluCalAmountCommon(const Call& call_node);

/*! \brief Count calculation amount of Eltwise layers. */
AiTraceDataFrame GetEltwiseCalAmountCommon(const Call& call_node, const std::string& op_type);

/*! \brief Count calculation amount of cal-free layer. */
AiTraceDataFrame GetZeroCalAmountCommon(const Call& call_node);

/*! \brief Get common memory info. */
AiTraceDataFrame GetMemoryCommon(const Call& call_node);

/*! \brief Calculation Amount Indicator
 *
 * Hold calculation amount indicator for trace data and convert it into target data structure.
 *
 */
class CalculationAmontIndicator {
 public:
  /*! \brief An empty calculation amount indictor */
  CalculationAmontIndicator() : fused_mul_add(0), mul(0), div(0), add(0), sub(0), exp(0), comp(0) {}

  /*! \brief Construct calculation amount indictor from unordered_map */
  explicit CalculationAmontIndicator(std::unordered_map<std::string, int64_t> data) {
    if (data.size() == 0) {
      new (this) CalculationAmontIndicator();
    } else {
      fused_mul_add = GetMapValues(data, "fused_mul_add", 0);
      mul = GetMapValues(data, "mul", 0);
      div = GetMapValues(data, "div", 0);
      add = GetMapValues(data, "add", 0);
      sub = GetMapValues(data, "sub", 0);
      exp = GetMapValues(data, "exp", 0);
      comp = GetMapValues(data, "comp", 0);
    }
  }

  ~CalculationAmontIndicator() {}

  /*! \brief Convert indicator data into aitrace frame */
  AiTraceDataFrame GetIndicatorMap() {
    AiTraceDataFrame res;
    Map<String, ObjectRef> inner_map;

    inner_map.Set("fused_mul_add", Integer(fused_mul_add));
    inner_map.Set("mul", Integer(mul));
    inner_map.Set("div", Integer(div));
    inner_map.Set("add", Integer(add));
    inner_map.Set("sub", Integer(sub));
    inner_map.Set("exp", Integer(exp));
    inner_map.Set("comp", Integer(comp));

    res.Set("calculation_amount", inner_map);
    return res;
  }

 public:
  int64_t fused_mul_add;
  int64_t mul;
  int64_t div;
  int64_t add;
  int64_t sub;
  int64_t exp;
  int64_t comp;
};

/*! \brief Memory information Indicator
 *
 * Hold Memory indicators for trace data and convert it into target data structure.
 *
 */
class MemoryIndicator {
 public:
  /*! \brief An empty memory indictor */
  MemoryIndicator() : params(0), output(0) {}
  ~MemoryIndicator() {}

  /*! \brief Convert indicator data into aitrace frame */
  AiTraceDataFrame GetIndicatorMap() {
    AiTraceDataFrame res;
    Map<String, ObjectRef> inner_map;

    inner_map.Set("params", Integer(params));
    inner_map.Set("output", Integer(output));

    res.Set("memory", inner_map);
    return res;
  }

 public:
  /*! The number of parameters in an op. */
  int64_t params;
  /*! The size of outputs in an op. */
  int64_t output;
};

class AddProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class AvgPool2dProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class BatchFlattenProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class BatchNormProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class BiasAddProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class ConcatenateProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class Conv2dProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class Conv2dTranposeProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class DenseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class DropoutProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class GlobalAvgPool2dProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class GlobalMaxPool2dProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class LRNProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class L2NormalizeNProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class MaximumProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class MaxPool2dProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class MaxPool2dLocationProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class MaxPool2dWithArgmaxProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class MultiplyProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class PreluProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class ProposalProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class PsroipoolingProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class ReluProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class ReshapeProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class RoiPoolProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class SigmoidProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class SoftmaxProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class SplitProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class StridedSliceProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class TanhProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class TransposeProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class UnpoolingProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class UpsamplingProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

}  // namespace aitrace
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ANALYSIS_GET_AITRACE_DATA_H_
