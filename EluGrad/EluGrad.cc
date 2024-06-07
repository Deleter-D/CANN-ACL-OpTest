#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "common/file_helper.h"
#include "common/generator.h"
#include "common/logging.h"
#include "common/nputensor.h"

#define ACL_CALL(msg) CHECK_EQ(reinterpret_cast<aclError>(msg), ACL_SUCCESS)

int main() {
  // Init
  ACL_CALL(aclInit(nullptr));
  ACL_CALL(aclrtSetDevice(0));

  // op type
  const std::string op_type = "EluGradV2";
  auto dtype_acl = ACL_FLOAT;
  using dtype = float;
  int rows = 30;
  int cols = 5;

  // input
  const std::vector<int64_t> dout_dims{rows * cols};
  const std::vector<dtype> dout_data = LoadTensorFromFile<dtype>(
      "/work/CANN-ACL-OpTest/EluGrad/dout.bin", rows * cols);
  std::cout << "dout_data: " << std::endl;
  for (int i = 0; i < cols; i++) {
    std::cout << dout_data[i] << " ";
  }
  std::cout << std::endl;
  auto dout = new npuTensor<dtype>(dtype_acl,
                                   dout_dims.size(),
                                   dout_dims.data(),
                                   ACL_FORMAT_NCHW,
                                   dout_data.data());

  const std::vector<int64_t> out_dims{rows * cols};
  const std::vector<dtype> out_data = LoadTensorFromFile<dtype>(
      "/work/CANN-ACL-OpTest/EluGrad/out.bin", rows * cols);
  std::cout << "out_data: " << std::endl;
  for (int i = 0; i < cols; i++) {
    std::cout << out_data[i] << " ";
  }
  std::cout << std::endl;
  auto out = new npuTensor<dtype>(dtype_acl,
                                  out_dims.size(),
                                  out_dims.data(),
                                  ACL_FORMAT_NCHW,
                                  out_data.data());

  // inputs
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(dout->desc);
  input_descs.emplace_back(out->desc);
  input_buffers.emplace_back(dout->buffer);
  input_buffers.emplace_back(out->buffer);

  // output
  const std::vector<int64_t> dx_dims{rows * cols};
  auto dx = new npuTensor<dtype>(
      dtype_acl, dx_dims.size(), dx_dims.data(), ACL_FORMAT_NCHW, nullptr);
  std::vector<dtype> dx_h(rows * cols, 0);

  // outputs
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(dx->desc);
  output_buffers.emplace_back(dx->buffer);

  // attributes
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrInt(attr, "alpha", 1.0));

  // create stream
  aclrtStream stream = nullptr;
  ACL_CALL(aclrtCreateStream(&stream));

  // run operator
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

  //   dx->Print("output");
  auto dx_size = aclGetTensorDescSize(dx->desc);
  ACL_CALL(aclrtMemcpy(dx_h.data(),
                       dx_size,
                       dx->device_ptr,
                       dx_size,
                       ACL_MEMCPY_DEVICE_TO_HOST));
  dx->Destroy();

  std::cout << "npu result:\t";
  for (int i = 0; i < cols; i++) {
    std::cout << dx_h[i] << " ";
  }
  std::cout << std::endl;

  std::vector<dtype> dx_from_file = LoadTensorFromFile<dtype>(
      "/work/CANN-ACL-OpTest/EluGrad/dx.bin", rows * cols);
  std::cout << "host result:\t";
  for (int i = 0; i < cols; i++) {
    std::cout << dx_from_file[i] << " ";
  }
  std::cout << std::endl;

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}
