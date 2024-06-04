#pragma once

#include <vector>
#include <random>

template<typename T>
std::vector<T> generateRandomVector(size_t length, T min_value, T max_value, unsigned int seed = std::random_device{}()) {
    std::vector<T> random_vector(length);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> dis(min_value, max_value);

    for (size_t i = 0; i < length; ++i) {
        random_vector[i] = dis(gen);
    }

    return random_vector;
}
