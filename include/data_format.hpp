#pragma once



#include <opencv2/core.hpp>
#include <vector>
#include <memory>
using data_type=double;

class Tensor3D
{
public:
    const int C,H,W;
    data_type* data;
    std::string name;
    //形状为C*H*W的3D张量
    Tensor3D(const int _C,const int _H,const int _W,const std::string name="pipeline")
        :C(C),H(H),W(W),name(std::move(name)){}
    // 形状 C x H x W, 分配内存
    Tensor3D(const std::tuple<int,int,int>& shape,const std::string _name="pipeline")
        :C(std::get<0>(shape)),H(std::get<1>(shape)),W(std::get<2>(shape)),
        data(new data_type[C*H*W]),name(std::move(_name)){}
    // 形状 length x 1 x 1, 此时length = C, 全连接层用得到
    Tensor3D(const int length,const std::string _name="pipeline")
        :Tensor3D({length,1,1},_name){}
    
    //从图像指针中获取内容
    void read_from_opencv_mat(const uchar* const img_ptr);
    //清零
    void set_zero();
    //找最大值
    data_type max() const;
    //找对最小值
    data_type min() const;
    //找最大值的位置
    std::tuple<int,int,int> max_position() const;
    //找最小值的位置
    std::tuple<int,int,int> min_position() const;

    int argmax() const;
    int argmin() const;

    void div(const data_type& times);

    void normalize(const std::vector<data_type> mean={0.406, 0.456, 0.485}, const std::vector<data_type> std_div={0.225, 0.224, 0.229});


    cv::Mat opencv_mat(const int CH=3)  const;

    int get_length() const ;

    std::tuple<int,int,int> get_shape() const;
    // 打印这个 Tensor 的形状
    void print_shape() const;
    // 打印这个 Tensor 在第 _C 个通道的内容
    void print(const int _C) const;

    std::shared_ptr<Tensor3D>rot180()   const;
    std::shared_ptr<Tensor3D>pad(const int padding=1) const;

    ~Tensor3D()noexcept;
};

using tensor = std::shared_ptr<Tensor3D>;