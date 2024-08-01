//C++
#include <vector>
#include <memory>
#include <iostream>
#include <filesystem>
// self
#include "utils.hpp"
#include "metrics.hpp"
#include "architectures.hpp"


int main(){
    std::setbuf(stdout, NULL);
    std::cout<<"opencv  :   "<<CV_VERSION<<std::endl;

    using namespace architectures;

    const int train_batch_size = 10;
    const int valid_batch_size = 2;
    const int test_batch_size = 2;

    assert(train_batch_size >= valid_batch_size && train_batch_size >= test_batch_size);
    
    const std::tuple<int,int,int>image_size({200,200,3});
    const std::filesystem::path dataset_dir = std::filesystem::path(IMAGE_DIR)/"dataset";
    


    const std::vector<std::string>categories({"0","1","2","3","4","5","6","7","8","9"});

    //获取图片
    auto dataset=pipeline::get_images_for_classification(dataset_dir,categories);

    //构造数据流
    pipeline::DataLoader train_loader(dataset["train"],train_batch_size,false,true,image_size);
    pipeline::DataLoader valid_loader(dataset["valid"], valid_batch_size, false, false, image_size);

    // 定义网络结构
    const int num_classes = categories.size(); // 分类的数目
    AlexNet network(num_classes, false);
}