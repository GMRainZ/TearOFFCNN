#ifndef CNN_METRICS_HPP
#define CNN_METRICS_HPP

#include <vector>

class ClassificationEvaluator{
private:
    int correct_num=0;
    int sample_num=0;
public:
    ClassificationEvaluator()=default;

    /**
     * @brief 计算分类的准确率
     * 
     * @param predict 
     * @param labels 
     */
    void compute(const std::vector<int>&predict, const std::vector<int>& labels);

    /**
     * @brief Get the accuracy object
     * 
     * @return double 
     */
    double get_accuracy() const;

    /**
     * @brief 重置准确率
     * 
     */
    void reset();
};


#endif // CNN_METRICS_HPP