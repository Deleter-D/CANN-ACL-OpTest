#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "common/generator.h"
#include "common/logging.h"
#include "common/nputensor.h"

#define ACL_CALL(msg) CHECK_EQ(reinterpret_cast<aclError>(msg), ACL_SUCCESS)

int main() {
  // Init
  ACL_CALL(aclInit(nullptr));
  ACL_CALL(aclrtSetDevice(0));

  // op type
  const std::string op_type = "Elu";
  auto dtype_acl = ACL_FLOAT;
  using dtype = float;
  int rows = 2;
  int cols = 3;

  // input
  const std::vector<int64_t> input_dims{rows * cols};
  const std::vector<dtype> input_data =
      generateRandomVector<dtype>(rows * cols, -1, 1);
  auto input = new npuTensor<dtype>(dtype_acl,
                                    input_dims.size(),
                                    input_dims.data(),
                                    ACL_FORMAT_NCHW,
                                    input_data.data());

  // inputs
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input->desc);
  input_buffers.emplace_back(input->buffer);

  // output
  const std::vector<int64_t> output_dims{rows * cols};
  auto output = new npuTensor<dtype>(dtype_acl,
                                     output_dims.size(),
                                     output_dims.data(),
                                     ACL_FORMAT_NCHW,
                                     nullptr);
  std::vector<dtype> output_h(rows * cols, 0);

  // outputs
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output->desc);
  output_buffers.emplace_back(output->buffer);

  // attributes
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrInt(attr, "alpha", 0.4));
  ACL_CALL(aclopSetAttrInt(attr, "scale", 1.0));
  ACL_CALL(aclopSetAttrInt(attr, "input_scale", 1.0));

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

  output->Print("output");
  auto out_size = aclGetTensorDescSize(output->desc);
  ACL_CALL(aclrtMemcpy(output_h.data(),
                       out_size,
                       output->device_ptr,
                       out_size,
                       ACL_MEMCPY_DEVICE_TO_HOST));
  output->Destroy();

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << output_h[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}
