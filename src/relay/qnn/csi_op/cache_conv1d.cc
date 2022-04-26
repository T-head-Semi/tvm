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
 * \file src/relay/qnn/op/matmul.cc
 * \brief Property def of qnn matmul operator.
 */

#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include "../../op/nn/nn.h"
#include "../utils.h"

namespace tvm {
namespace relay {
namespace qnn {

// relay.custom.cache_conv1d
TVM_REGISTER_NODE_TYPE(QnnCSICacheConv1DAttrs);

bool QnnCSICacheConv1DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                          const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr) return false;
  static const Layout kNCW("NCW");
  static const Layout kOIW("OIW");

  const QnnCSICacheConv1DAttrs* param = attrs.as<QnnCSICacheConv1DAttrs>();
  ICHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = tir::BijectiveLayout(in_layout, kNCW);
  ICHECK(trans_in_layout.defined())
      << "Conv only support input layouts that are convertible from NCW."
      << " But got " << in_layout;

  const auto trans_kernel_layout = tir::BijectiveLayout(kernel_layout, kOIW);
  ICHECK(trans_kernel_layout.defined())
      << "Conv only support kernel layouts that are convertible from OIW."
      << " But got " << kernel_layout;

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = tir::BijectiveLayout(out_layout, kNCW);
  ICHECK(trans_out_layout.defined())
      << "Conv only support output layouts that are convertible from NCW."
      << " But got " << out_layout;

  Array<IndexExpr> dshape_nwc = trans_in_layout.ForwardShape(data->shape);

  IndexExpr channels, dilated_ksize;
  // infer weight if the kernel_size and channels are defined
  if (param->kernel_size.defined() && param->channels.defined()) {
    Array<IndexExpr> wshape;

    wshape = {{param->channels, dshape_nwc[2], param->kernel_size[0]}};

    wshape = trans_kernel_layout.BackwardShape(wshape);
    channels = param->channels;
    dilated_ksize = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
    DataType weight_dtype = data->dtype;
    if (weight != nullptr) {
      weight_dtype = weight->dtype;
    }
    // assign result to reporter
    reporter->Assign(types[1], TensorType(wshape, weight_dtype));
  } else {
    // use weight to infer the conv shape.
    if (weight == nullptr) return false;
    auto wshape = trans_kernel_layout.ForwardShape(weight->shape);
    if (param->kernel_size.defined()) {
      // check the size
      ICHECK(reporter->AssertEQ(param->kernel_size[0], wshape[2]))
          << "Conv1D: shape of weight is inconsistent with kernel_size, "
          << " kernel_size=" << param->kernel_size << " wshape=" << wshape;
    }
    if (param->channels.defined()) {
      ICHECK(reporter->AssertEQ(param->channels, wshape[0]))
          << "Conv1D: shape of weight is inconsistent with channels, "
          << " channels=" << param->channels << " wshape=" << wshape;
    }
    if (!dshape_nwc[2].as<tir::AnyNode>() && !wshape[1].as<tir::AnyNode>()) {
      ICHECK(reporter->AssertEQ(dshape_nwc[2], wshape[1]));
    }
    channels = wshape[0];
    dilated_ksize = 1 + (wshape[2] - 1) * param->dilation[0];
  }
  // dilation
  Array<IndexExpr> oshape({dshape_nwc[0], channels, 0});

  if (!dshape_nwc[1].as<tir::AnyNode>()) {
    oshape.Set(2, param->cache_shape[2] + dshape_nwc[1]);
  } else {
    oshape.Set(2, dshape_nwc[1]);
  }

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  oshape = trans_out_layout.BackwardShape(oshape);
  // assign output type
  reporter->Assign(types[3], TensorType(oshape, out_dtype));
  return true;
}

Expr MakeQnnCSICacheConv1D(Expr data, Expr weight, Expr bias, Array<Integer> cache_shape,
                           Array<IndexExpr> strides, Array<IndexExpr> padding,
                           Array<IndexExpr> dilation, int groups, IndexExpr channels,
                           Array<IndexExpr> kernel_size, String data_layout, String kernel_layout,
                           String out_layout, DataType out_dtype, Array<Array<IndexExpr>> q_params,
                           String layer_name) {
  auto attrs = make_object<QnnCSICacheConv1DAttrs>();
  attrs->cache_shape = std::move(cache_shape);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);
  const Op& op = Op::Get("qnn.csi.cache_conv1d");
  return Call(op, {data, weight, bias}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.cache_conv1d")
    .describe(R"code(
    custom fusion ops.
    (Cache)    Input
      |          |
    Gather   Transpose
       \        /
         Concat               --> CacheConv1d
           |
         Conv1d
           |
        BiasAdd

)code" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .set_attrs_type<QnnCSICacheConv1DAttrs>()
    .add_argument("data", "Tensor", "The input data tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .add_argument("bias", "Tensor", "The bias tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSICacheConv1D", QnnCSICacheConv1DRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSICacheConv1D").set_body_typed(MakeQnnCSICacheConv1D);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
