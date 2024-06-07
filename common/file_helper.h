#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// 模板函数声明，用于读取张量数据
template <typename T>
std::vector<T> LoadTensorFromFile(const std::string& file_path,
                                  size_t num_elements) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Unable to open file: " << file_path << std::endl;
    return std::vector<T>();
  }

  // 确保vector有足够的空间
  std::vector<T> tensor_data(num_elements, 0);

  // 读取二进制数据到vector中
  file.read(reinterpret_cast<char*>(tensor_data.data()),
            num_elements * sizeof(T));
  if (!file) {
    std::cerr << "Error reading file: " << file_path << std::endl;
    return std::vector<T>();
  }

  file.close();
  return tensor_data;
}

template <typename T>
void SaveTensorToFile(const std::string& file_path,
                      const T* tensor_data,
                      size_t num_elements) {
  std::ofstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Unable to open file: " << file_path << std::endl;
    return;
  }

  // 写入二进制数据到文件中
  file.write(reinterpret_cast<const char*>(tensor_data),
             num_elements * sizeof(T));
  if (!file) {
    std::cerr << "Error writing to file: " << file_path << std::endl;
    return;
  }

  file.close();
}