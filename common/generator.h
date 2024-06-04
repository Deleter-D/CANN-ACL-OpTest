#pragma once

#include <random>
#include <type_traits>
#include <vector>

template <typename T>
std::vector<T> generateRandomVector(
    size_t length,
    T min_value,
    T max_value,
    unsigned int seed = std::random_device{}()) {
  std::vector<T> random_vector(length);
  std::mt19937 gen(seed);

  using dist_type =
      typename std::conditional<std::is_floating_point<T>::value,
                                std::uniform_real_distribution<T>,
                                std::uniform_int_distribution<T>>::type;

  dist_type dis(min_value, max_value);

  for (size_t i = 0; i < length; ++i) {
    random_vector[i] = dis(gen);
  }

  return random_vector;
}