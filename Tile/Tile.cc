#include <iostream>
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
  const std::string op_type = "Tile";
  // input - x
  const std::vector<int64_t> x_dims{2, 3};
  bool x_data[6] = {false};
  x_data[1] = true;
  x_data[2] = true;
  x_data[3] = true;
  // std::vector<bool> x_data(6);
  // std::iota(x_data.begin(), x_data.end(), 0);
  // input - muls
  const std::vector<int64_t> m_dims{2};
  std::vector<int64_t> m_data{1, 2};
  // output
  const std::vector<int64_t> y_dims{2, 6};

  // inputs
  auto input_x = new npuTensor<bool>(
      ACL_BOOL, x_dims.size(), x_dims.data(), ACL_FORMAT_ND, x_data);
  auto input_m = new npuTensor<int64_t>(ACL_INT64,
                                        m_dims.size(),
                                        m_dims.data(),
                                        ACL_FORMAT_ND,
                                        m_data.data(),
                                        memType::HOST);
  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input_x->desc);
  input_descs.emplace_back(input_m->desc);
  input_buffers.emplace_back(input_x->buffer);
  input_buffers.emplace_back(input_m->buffer);

  // output
  auto output_y = new npuTensor<bool>(
      ACL_BOOL, y_dims.size(), y_dims.data(), ACL_FORMAT_ND, nullptr);
  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output_y->desc);
  output_buffers.emplace_back(output_y->buffer);

  // attributes
  auto attr = aclopCreateAttr();

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
  // input_x->Print("x");
  // input_m->Print("m");
  // output_y->Print("y");

  // destroy
  input_x->Destroy();
  input_m->Destroy();
  output_y->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}