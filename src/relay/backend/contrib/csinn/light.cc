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
 * \file src/relay/backend/contrib/csinn/light.cc
 * \brief Implementation of CSINN light codegen APIs.
 */

#include "light.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

using namespace tvm::relay::qnn;
namespace tvm {
namespace relay {
namespace contrib {
using namespace backend;

void CodegenLight::VisitExpr_(const CallNode* call) {
  /* Get the arguments for various CSINN kernels. */
  /* QNN op */
  if (first_visit_expr) {
    first_visit_expr = false;
    Output output;
    output.call = call;
    output_list_.push_back(output);
  }
  if (IsOp(call, "qnn.csi.add")) {
    DisoOp(call, "add");
  } else if (IsOp(call, "qnn.csi.avgpool2d")) {
    AvgPool2d(call);
  } else if (IsOp(call, "qnn.csi.batch_to_space_nd")) {
    BatchToSpaceND(call);
  } else if (IsOp(call, "qnn.csi.bias_add")) {
    DisoOp(call, "add");
  } else if (IsOp(call, "qnn.csi.clip")) {
    Clip(call);
  } else if (IsOp(call, "qnn.csi.concatenate")) {
    Concat(call);
  } else if (IsOp(call, "qnn.csi.conv2d")) {
    Conv2d(call, "conv2d");
  } else if (IsOp(call, "qnn.csi.conv2d_relu")) {
    Conv2d(call, "conv2d_relu");
  } else if (IsOp(call, "qnn.csi.conv2d_relu6")) {
    Conv2d(call, "conv2d_relu6");
  } else if (IsOp(call, "qnn.csi.deconv2d")) {
    DeConv2d(call);
  } else if (IsOp(call, "qnn.csi.dense")) {
    Dense(call);
  } else if (IsOp(call, "qnn.csi.depth_to_space")) {
    DepthToSpace(call);
  } else if (IsOp(call, "qnn.csi.exp")) {
    Unary(call, "exp");
  } else if (IsOp(call, "qnn.csi.flatten")) {
    Flatten(call);
  } else if (IsOp(call, "qnn.csi.global_avgpool2d")) {
    GlobalAvgPool2d(call);
  } else if (IsOp(call, "qnn.csi.global_maxpool2d")) {
    GlobalMaxPool2d(call);
  } else if (IsOp(call, "qnn.csi.leaky_relu")) {
    LeakyRelu(call);
  } else if (IsOp(call, "qnn.csi.lrn")) {
    LRN(call);
  } else if (IsOp(call, "qnn.csi.maxpool2d")) {
    MaxPool2d(call);
  } else if (IsOp(call, "qnn.csi.maxpool2d_locat")) {
    MaxPool2dLocat(call);
  } else if (IsOp(call, "qnn.csi.maxpool2d_with_argmax")) {
    Maxpool2dWithArgmax(call);
  } else if (IsOp(call, "qnn.csi.mean")) {
    Reduce(call, "mean", cfg->dtype_weight);
  } else if (IsOp(call, "qnn.csi.mul")) {
    DisoOp(call, "mul");
  } else if (IsOp(call, "qnn.csi.div")) {
    DisoOp(call, "div");
  } else if (IsOp(call, "qnn.csi.minimum")) {
    DisoOp(call, "minimum");
  } else if (IsOp(call, "qnn.csi.maximum")) {
    DisoOp(call, "maximum");
  } else if (IsOp(call, "qnn.csi.subtract")) {
    DisoOp(call, "sub");
  } else if (IsOp(call, "qnn.csi.prelu")) {
    PRelu(call);
  } else if (IsOp(call, "qnn.csi.pad")) {
    Pad(call);
  } else if (IsOp(call, "qnn.csi.proposal")) {
    Proposal(call);
  } else if (IsOp(call, "qnn.csi.psroipooling")) {
    PSROIPool(call);
  } else if (IsOp(call, "qnn.csi.relu")) {
    Relu(call);
  } else if (IsOp(call, "qnn.csi.relu6")) {
    Relu6(call);
  } else if (IsOp(call, "qnn.csi.reshape")) {
    Reshape(call);
  } else if (IsOp(call, "qnn.csi.roipooling")) {
    ROIPool(call);
  } else if (IsOp(call, "qnn.csi.sigmoid")) {
    Sigmoid(call);
  } else if (IsOp(call, "qnn.csi.softmax")) {
    Softmax(call);
  } else if (IsOp(call, "qnn.csi.space_to_batch_nd")) {
    SpaceToBatchND(call);
  } else if (IsOp(call, "qnn.csi.split")) {
    Split(call);
  } else if (IsOp(call, "qnn.csi.squeeze")) {
    Squeeze(call);
  } else if (IsOp(call, "qnn.csi.strided_slice")) {
    StridedSlice(call);
  } else if (IsOp(call, "qnn.csi.transpose")) {
    Transpose(call);
  } else if (IsOp(call, "qnn.csi.unpooling")) {
    UnPool2d(call);
  } else if (IsOp(call, "qnn.csi.upsampling")) {
    UpSampling(call);
  } else {
    std::cerr << "light unsupported op: " << AsText(call->op, false) << "\n";
    exit(-1);
  }
}
void CodegenLight::EmitHeader(void) {
  std::ostringstream t0;
  func_def_.OneLine("#include <shl_pnna.h>");
  func_def_.NewLine();
}

void CodegenLight::EmitSessionSetup(void) {
  std::ostringstream t0;
  t0 << "void *" << ext_func_id_ << "_(";
  t0 << "char *params_base) {";
  func_def_.OneLine(t0);
  func_def_.EnterScope();

  func_def_.OneLine("struct csinn_session *sess = csinn_alloc_session();");
  SessionRunMode();
  ModelBinarySave();
  t0 << "sess->base_api = " << target_name_ << ";";
  func_def_.OneLine(t0);
  t0 << "sess->base_dtype = " << base_dtype_ << ";";
  func_def_.OneLine(t0);
  if (debug_level_ == "INFO") {
    func_def_.OneLine("sess->debug_level = CSINN_DEBUG_LEVEL_INFO;");
  }
  func_def_.OneLine("csinn_session_init(sess);");

  t0 << "csinn_set_input_number(" << ext_func_args_.size() << ", sess);";
  func_def_.OneLine(t0);
  t0 << "csinn_set_output_number(" << output_list_.size() << ", sess);";
  func_def_.OneLine(t0);

  func_def_.NewLine();
  for (uint32_t i = 0; i < ext_func_args_.size(); i++) {
    std::string in_name = CodegenCSINN::replace(ext_func_args_[i]->name_hint());
    std::ostringstream t1;
    t1 << in_name << "->mtype = " << GetCSINNMemoryType(input_memory_type[i]);
    func_def_.PushDecl(t1);
    t1 << "csinn_set_tensor_entry(" << in_name << ", sess)";
    func_def_.PushDecl(t1);
    t1 << "csinn_set_input(" << i << ", " << in_name << ", sess)";
    func_def_.PushDecl(t1);
  }

  func_def_.BufToCode();

  int output_index = 0;
  // emit normal outputs
  for (uint32_t i = 0; i < output_list_.size(); i++) {
    if (!output_list_[i].is_const) {
      string output_name = output_list_[i].name;
      t0 << "csinn_set_output(" << output_index++ << ", " << output_name << ", sess);";
      func_def_.OneLine(t0);
    }
  }

  // emit constant outputs
  for (uint32_t i = 0; i < output_list_.size(); i++) {
    if (output_list_[i].is_const) {
      t0 << output_list_[i].name << "->name = "
         << "\"" << output_list_[i].name << "\";";
      func_def_.OneLine(t0);
      t0 << output_list_[i].name << "->dtype = CSINN_DTYPE_FLOAT32;";
      func_def_.OneLine(t0);
      t0 << output_list_[i].name << "->is_const = 1;";
      func_def_.OneLine(t0);
      t0 << "csinn_set_output(" << output_index++ << ", " << output_list_[i].name << ", sess);";
      func_def_.OneLine(t0);
    }
  }

  auto ctx = transform::PassContext::Current();
  auto opt = ctx->GetConfig<CSINNConfig>("relay.ext.csinn.options");
  auto opt_cfg = opt.value();

  double fix_height = opt_cfg->light_input_fix_height;
  double fix_width = opt_cfg->light_input_fix_width;
  if (fix_height != 0) {
    t0 << "shl_pnna_set_input_strides(sess, 1, " << fix_height << " ," << fix_width << ");";
    func_def_.OneLine(t0);
  }

  func_def_.NewLine();
  func_def_.OneLine("csinn_session_setup(sess);");
  func_def_.OneLine("return sess;");
  func_def_.ExitScope();
  func_def_.OneLine("}");
}

void CodegenLight::GetSymScale(float min_value, float max_value, int bits, Qinfo* qinfo) {
  int valid_range = std::pow(2, bits - 1) - 1;
  float abs_max = std::max(std::abs(min_value), std::abs(max_value));
  float scale = valid_range / abs_max;
  int exponent;
  frexp(scale, &exponent);
  qinfo->scale = 1.0f / std::pow(2, exponent - 1);
  qinfo->zero_point = 0;
  qinfo->max = abs_max;
  qinfo->min = -abs_max;
}

void CodegenLight::EmitJitWrapper() {
  func_def_.NewLine();
  std::ostringstream t0;
  string in_dtype = cfg->dtype_input;
  string weight_dtype = cfg->dtype_weight;
  func_def_.NewLine();
  t0 << "int csinn_runtime_wrapper_(";
  t0 << "int64_t* arg_value, ";
  t0 << "int64_t* arg_type, ";
  t0 << "int64_t* arg_size, ";
  t0 << "int64_t* ret_vale, int64_t* ret_type_code) {";
  func_def_.OneLine(t0);

  func_def_.EnterScope();
  func_def_.OneLine("char *params_base = (char *)arg_value[2];");

  t0 << ext_func_id_ << "_(params_base);\n";

  func_def_.OneLine(t0);
  func_def_.OneLine("return 0;");
  func_def_.ExitScope();
  func_def_.OneLine("}");
}

void CodegenLight::EmitNBGSetup(void) {
  for (uint i = 0; i < output_list_.size(); i++) {
    if (!output_list_[i].is_const) {
      string output_name = output_list_[i].name;
      CSINNTensor* output_tensor = bm_graph.get_tensor(output_name);
      output_tensor->tensor->name = const_cast<char*>(output_name.c_str());
      bm_graph.set_output(output_tensor);
    }
  }
  for (uint i = 0; i < ext_func_args_.size(); i++) {
    std::string new_name = CodegenCSINN::replace(ext_func_args_[i]->name_hint());
    CSINNTensor* input_tensor = bm_graph.get_tensor(new_name);
    input_tensor->tensor->name = const_cast<char*>(new_name.c_str());
    bm_graph.set_input(input_tensor);
  }
  string q_scheme = cfg->quantization_scheme;
  if (q_scheme == "CSINN_QUANT_INT8_ASYM_W_SYM" || q_scheme == "CSINN_QUANT_INT8_ORIGINAL") {
    q_scheme = "CSINN_QUANT_INT8_ASYM";
  }
  bm_graph.sess->base_quant_type = CSINNTensorQuantStringToEnum(q_scheme);

  bm_graph.sess->base_api = CSINN_LIGHT;
  bm_graph.sess->base_dtype = CSINNTensorDtypeStringToEnum(base_dtype_);

  if (debug_level_ == "INFO") {
    bm_graph.sess->debug_level = CSINN_DEBUG_LEVEL_INFO;
  }

  auto ctx = transform::PassContext::Current();
  auto opt = ctx->GetConfig<CSINNConfig>("relay.ext.csinn.options");
  auto opt_cfg = opt.value();

  double fix_height = opt_cfg->light_input_fix_height;
  double fix_width = opt_cfg->light_input_fix_width;
  if (fix_height != 0) {
    /* rewrite csinn_import_binary_model to replace weak define in SHL */
    std::ostringstream t0;
    func_def_.NewLine();
    t0 << "struct csinn_session *csinn_import_binary_model(char *bm_addr) {";
    func_def_.OneLine(t0);
    func_def_.EnterScope();

    t0 << "struct shl_binary_model_section_info *sinfo = "
       << "(struct shl_binary_model_section_info *)(bm_addr + 4096);";
    func_def_.OneLine(t0);
    t0 << "struct csinn_session *bm_sess = "
       << "(struct csinn_session *)(bm_addr + sinfo->sections->info_offset * 4096);";
    func_def_.OneLine(t0);
    t0 << "struct csinn_session *sess = csinn_alloc_session();";
    func_def_.OneLine(t0);
    t0 << "shl_bm_session_load(sess, bm_sess);";
    func_def_.OneLine(t0);
    t0 << "sess->model.bm_addr = bm_addr + sinfo->sections->graph_offset * 4096;";
    func_def_.OneLine(t0);
    t0 << "sess->model.bm_size = sinfo->sections->graph_size;";
    func_def_.OneLine(t0);

    t0 << "shl_pnna_set_input_strides(sess, 1, " << fix_height << " ," << fix_width << ");";
    func_def_.OneLine(t0);

    t0 << "csinn_load_binary_model(sess);";
    func_def_.OneLine(t0);
    t0 << "return sess;";
    func_def_.OneLine(t0);
    func_def_.ExitScope();
    func_def_.OneLine("}");
  }
}

string CodegenLight::EmitGraph(void) {
  EmitVersion();
  EmitHeader();
  EmitSessionSetup();
  if (model_save == "save_only") {
    EmitJitWrapper();
  } else {
    EmitSessionRun();
    EmitNBGSetup();
  }
  DumpConstant();
  DumpGraphInfo();
  return func_def_.str();
}

void CodegenLight::ModelBinarySave() {
  std::ostringstream t0;
  std::string quantization_scheme = cfg->quantization_scheme == "CSINN_QUANT_INT8_ASYM_W_SYM"
                                        ? "CSINN_QUANT_INT8_ASYM"
                                        : cfg->quantization_scheme;
  t0 << "sess->base_quant_type = " << quantization_scheme << ";";
  func_def_.OneLine(t0);
  if (model_save == "save_only") {
    t0 << "sess->model.save_mode = CSINN_SAVE_ONLY;";
    func_def_.OneLine(t0);
  }
  if (model_priority >= 0 && model_priority <= 2) {
    t0 << "sess->model.priority = " << to_string(model_priority) << ";";
  } else {
    std::cerr << "light unsupported priority " << to_string(model_priority) << "\n";
  }
  func_def_.OneLine(t0);
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
