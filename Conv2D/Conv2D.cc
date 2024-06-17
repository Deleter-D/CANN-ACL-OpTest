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
  const std::string op_type = "Conv2D";
  auto dtype_acl = ACL_FLOAT;
  using dtype = float;

  // input
  const std::vector<int64_t> input_dims{1, 3, 224, 224};
  const std::vector<dtype> input_data = LoadTensorFromFile<dtype>(
      "/work/CANN-ACL-OpTest/Conv2D/input.bin",
      std::accumulate(
          input_dims.begin(), input_dims.end(), 1, std::multiplies<int>()));
  auto input = new npuTensor<dtype>(dtype_acl,
                                    input_dims.size(),
                                    input_dims.data(),
                                    ACL_FORMAT_NCHW,
                                    input_data.data());

  const std::vector<int64_t> filter_dims{32, 3, 3, 3};
  const std::vector<dtype> filter_data = LoadTensorFromFile<dtype>(
      "/work/CANN-ACL-OpTest/Conv2D/filter.bin",
      std::accumulate(
          filter_dims.begin(), filter_dims.end(), 1, std::multiplies<int>()));
  auto filter = new npuTensor<dtype>(dtype_acl,
                                     filter_dims.size(),
                                     filter_dims.data(),
                                     ACL_FORMAT_NCHW,
                                     filter_data.data());

  // inputs
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input->desc);
  input_descs.emplace_back(filter->desc);
  input_buffers.emplace_back(input->buffer);
  input_buffers.emplace_back(filter->buffer);

  // output
  const std::vector<int64_t> output_dims{1, 32, 112, 112};
  const int64_t output_count = std::accumulate(
      output_dims.begin(), output_dims.end(), 1, std::multiplies<int>());
  auto output = new npuTensor<dtype>(dtype_acl,
                                     output_dims.size(),
                                     output_dims.data(),
                                     ACL_FORMAT_NCHW,
                                     nullptr);
  std::vector<dtype> output_h(output_count, 0);

  // outputs
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output->desc);
  output_buffers.emplace_back(output->buffer);

  // attributes
  auto attr = aclopCreateAttr();
  const std::vector<int64_t> strides{1, 1, 2, 2};
  const std::vector<int64_t> paddings{1, 1, 1, 1};
  ACL_CALL(aclopSetAttrListInt(attr, "strides", 4, strides.data()));
  ACL_CALL(aclopSetAttrListInt(attr, "pads", 4, paddings.data()));

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

  auto output_size = aclGetTensorDescSize(output->desc);
  ACL_CALL(aclrtMemcpy(output_h.data(),
                       output_size,
                       output->device_ptr,
                       output_size,
                       ACL_MEMCPY_DEVICE_TO_HOST));
  output->Destroy();

  SaveTensorToFile<dtype>("/work/CANN-ACL-OpTest/Conv2D/output_npu.bin",
                          output_h.data(),
                          output_count);
  std::cout << "npu result:\t";
  for (int i = 0; i < output_count; i++) {
    std::cout << output_h[i] << " ";
  }
  std::cout << std::endl;

  std::vector<dtype> dx_from_file = LoadTensorFromFile<dtype>(
      "/work/CANN-ACL-OpTest/Conv2D/output.bin", output_count);
  std::cout << "host result:\t";
  for (int i = 0; i < output_count; i++) {
    std::cout << dx_from_file[i] << " ";
  }
  std::cout << std::endl;

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}
