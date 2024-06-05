#include <algorithm>  // for copy
#include <iostream>
#include <iterator>  // for ostream_iterator
#include <numeric>
#include <vector>

#include "common/nputensor.h"

int main() {
  // Init
  ACL_CALL(aclInit(nullptr));
  ACL_CALL(aclrtSetDevice(0));

  // Get Run Mode - ACL_HOST
  aclrtRunMode runMode;
  ACL_CALL(aclrtGetRunMode(&runMode));
  std::string run_mode_str =
      (runMode == ACL_DEVICE) ? "ACL_DEVICE" : "ACL_HOST";
  std::cout << "aclrtRunMode is : " << run_mode_str << std::endl;

  // op type
  const std::string op_type = "Fill";
  // input - dims
  const std::vector<int64_t> dims_dims{2};
  const std::vector<int64_t> dims_data{10, 10};
  // input - value
  const std::vector<int64_t> value_dims{1};
  const std::vector<int64_t> value_data(1, 1);
  // output
  const std::vector<int64_t> output_dims{10, 10};

  // input - value
  auto input_dims = new npuTensor<int64_t>(ACL_INT64,
                                           dims_dims.size(),
                                           dims_dims.data(),
                                           ACL_FORMAT_ND,
                                           dims_data.data(),
                                           memType::DEVICE);
  auto input_value = new npuTensor<int64_t>(ACL_INT64,
                                            value_dims.size(),
                                            value_dims.data(),
                                            ACL_FORMAT_ND,
                                            value_data.data(),
                                            memType::DEVICE);

  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input_dims->desc);
  input_descs.emplace_back(input_value->desc);
  input_buffers.emplace_back(input_dims->buffer);
  input_buffers.emplace_back(input_value->buffer);

  // output - out
  auto output = new npuTensor<int64_t>(ACL_INT64,
                                       output_dims.size(),
                                       output_dims.data(),
                                       ACL_FORMAT_ND,
                                       nullptr);

  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output->desc);
  output_buffers.emplace_back(output->buffer);

  // Note: need to change data type first
  // int64_t input_value = static_cast<int64_t>(value);

  // std::cout << "input_value = " << input_value << std::endl;

  // attr
  auto attr = aclopCreateAttr();
  // ACL_CALL(aclopSetAttrFloat(attr, "value", input_value));
  // ACL_CALL(aclopSetAttrListInt(attr, "dims", output_dims.size(),
  // output_dims.data()));

  // create stream
  aclrtStream stream = nullptr;
  ACL_CALL(aclrtCreateStream(&stream));

  std::cout << "aclopCompileAndExecute : " << op_type << std::endl;
  ACL_CALL(aclopCompileAndExecute(op_type.c_str(),
                                  input_descs.size(),
                                  input_descs.data(),
                                  input_buffers.data(),
                                  output_descs.size(),
                                  output_descs.data(),
                                  output_buffers.data(),
                                  attr,
                                  ACL_ENGINE_SYS,
                                  ACL_COMPILE_SYS,
                                  NULL,
                                  stream));

  // sync and destroy stream
  ACL_CALL(aclrtSynchronizeStream(stream));
  ACL_CALL(aclrtDestroyStream(stream));

  // print output
  output->Print("y");

  // destroy - outputs
  input_dims->Destroy();
  input_value->Destroy();
  output->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}