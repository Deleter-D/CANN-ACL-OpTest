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
  const std::string op_type = "GatherNd";
  auto dtype_acl = ACL_FLOAT;
  auto index_type_acl = ACL_INT64;
  using dtype = float;
  using index_type = int64_t;
  int input_rows = 10;
  int input_cols = 128;
  int index_rows = 10;
  int index_cols = 64;

  // input
  const std::vector<int64_t> input_dims{input_rows, input_cols};
  const std::vector<dtype> input_data =
      generateRandomVector<dtype>(input_rows * input_cols, 0, 1);
  auto input = new npuTensor<dtype>(dtype_acl,
                                    input_dims.size(),
                                    input_dims.data(),
                                    ACL_FORMAT_NCHW,
                                    input_data.data());

  const std::vector<int64_t> index_dims{index_rows, index_cols, 2};
  const std::vector<index_type> index_data = generateRandomVector<index_type>(
      index_rows * index_cols, 0, input_cols - 1);
  std::vector<index_type> index_vec;
  for (auto i = 0; i < input_rows; i++) {
    for (auto j = 0; j < index_cols; j++) {
      index_vec.push_back(i);
      index_vec.push_back(index_data[i * index_cols + j]);
    }
  }
  auto index = new npuTensor<index_type>(index_type_acl,
                                         index_dims.size(),
                                         index_dims.data(),
                                         ACL_FORMAT_NCHW,
                                         index_vec.data());

  // inputs
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input->desc);
  input_descs.emplace_back(index->desc);
  input_buffers.emplace_back(input->buffer);
  input_buffers.emplace_back(index->buffer);

  // output
  const std::vector<int64_t> output_dims{index_rows * index_cols};
  auto output = new npuTensor<dtype>(dtype_acl,
                                     output_dims.size(),
                                     output_dims.data(),
                                     ACL_FORMAT_NCHW,
                                     nullptr);
  std::vector<dtype> output_h(index_rows * index_cols, 0);

  // outputs
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output->desc);
  output_buffers.emplace_back(output->buffer);

  // attributes
  auto attr = aclopCreateAttr();

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

  // for (int i = 0; i < index_rows; i++) {
  //   for (int j = 0; j < index_cols; j++) {
  //     std::cout << output_h[i * index_cols + j] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}
