#ifndef UTILS_HPP
#define UTILS_HPP

#include "data_format.hpp"

namespace Utils{

/**
 * @brief 计算softmax
 * 
 * @param input 
 */
std::vector<tensor>softmax(const std::vector<tensor>& input);

/**
 * @brief One_hot编码
 * 
 * @param labels 
 * @param num_classes 
 */
std::vector<tensor>one_hot(const std::vector<tensor>& labels,const int num_classes);

// 给输出概率 probs, 和标签 label 计算交叉熵损失, 返回损失值和回传的梯度
std::pair<data_type, std::vector<tensor> > cross_entroy_backward(
        const std::vector<tensor>& probs, const std::vector<tensor>& labels);

std::string float_to_string(const double value, const int precision = 6);

} // namespace Utils

#endif // UTILS_HPP