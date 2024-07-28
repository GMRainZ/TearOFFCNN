#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#include "pipeline.hpp"

#include <iostream>


namespace {
    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }

    cv::Mat rotate(cv::Mat& src, double angle) {
        // 抄自 https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
        // 角度最好在正负 15-75 之间, 这个程序还是有问题的
        cv::Point2f center((src.cols-1)/2.0, (src.rows-1)/2.0);
        cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
        rot.at<double>(0,2) += bbox.width/2.0 - src.cols/2.0;
        rot.at<double>(1,2) += bbox.height/2.0 - src.rows/2.0;
        cv::warpAffine(src, src, rot, bbox.size());
        return src;
    }
}


using namespace pipeline ;


void ImageAugmentor::make_augmentations(cv::Mat& origin,const bool show)
{
    std::shuffle(ops.begin(), ops.end(), this->shuffle_engine);
    for (const auto op : ops) {
        const double prob=engine(emancipate_engine);
        if( prob >= 1.0 - op.second)
        {
            if(op.first == "hflip")
            {
                cv::flip(origin, origin, 1);
            }
            else if(op.first == "vflip")
            {
                cv::flip(origin, origin, 0);
            }
            else if(op.first == "crop")
            {
                const int H=origin.rows;
                const int W=origin.cols;

                double cropRatio=0.7f+crop_ratio(crop_engine);
                
                std::uniform_int_distribution<int>_H_pos(0, H-int(H*cropRatio));
                std::uniform_int_distribution<int>_W_pos(0, W-int(W*cropRatio));

                origin=origin(cv::Rect(_W_pos(this->shuffle_engine), _H_pos(this->shuffle_engine), int(W*cropRatio), int(H*cropRatio))).clone();
            }
        }
        else if(op. first == "rotate")
        {
            double angle=rotate_ratio(rotate_engine);
            if(minus_engine(rotate_engine) & 1)angle=-angle;
            origin=rotate(origin, angle);
        }

        if(show)cv_show( origin, op.first.c_str());
    }
}

std::map<std::string,pipeline::list_type>pipeline::get_images_for_classification(
        const std::filesystem::path dataset_path,//数据集路径
        const std::vector<std::string> categories,//类别
        const std::pair<double,double>ratios//比例
){
    list_type all_images_list;//所有图片列表

    const int categories_num = categories.size();//类别数量

    //遍历每个类别
    for(int i=0;i<categories_num;++i)
    {
        const auto images_dir= dataset_path / categories[i];//图片路径
        assert(std::filesystem::exists(images_dir)&& std::string(images_dir.string()+"isn't exist").c_str());//检查路径是否存在
    
        //遍历图片路径
        auto walker=std::filesystem::directory_iterator(images_dir);
        for(const auto&item:walker)
            all_images_list.emplace_back(item.path().string(),i);//将图片路径和类别放入列表 
    }
    //打乱图像列表
    std::shuffle(all_images_list.begin(), all_images_list.end(), std::default_random_engine(212));//随机打乱列表

    //将数据集分为三部分
    const int total_size = all_images_list.size();//总大小
    assert(ratios.first>0 && ratios.second>0 && ratios.first+ratios.second<=1);//检查比例是否合法

    const int train_size = total_size * ratios.first;//训练集大小
    const int test_size = total_size * ratios.second;//测试集大小
    
    std::map<std::string, list_type> results;//图片分类

    results.emplace("train", list_type(all_images_list.begin(), all_images_list.begin() + train_size));//训练集
    results.emplace("test", list_type(all_images_list.begin() + train_size, all_images_list.begin() + train_size + test_size));//测试集
    results.emplace("valid", list_type(all_images_list.begin() + train_size + test_size, all_images_list.end()));//验证集

    std::cout<<"train size: "<<results["train"].size()<<", test size: " << results["test"].size()<<", val size: "<<results["valid"].size()<<std::endl;

    return results;
}

DataLoader::DataLoader(const list_type& _images_list, 
                const int _batch_size, const bool _augment, 
                const bool _shuffle, const int _seed, 
                const int _H, const int _W, const int _C):
                images_list(_images_list),
                batch_size(_batch_size),
                augment(_augment),
                shuffle(_shuffle),
                seed(_seed),
                H(_H),
                W(_W),
                C(_C){
    this->images_num=this->images_list.size();//图片数量
    this->buffer.reserve(this->batch_size);
    for(int i=0;i<this->batch_size;++i)
        this->buffer.emplace_back(new Tensor3D(C,H,W));
}

int DataLoader::length()const{return this->images_num;}

DataLoader::batch_type DataLoader::generate_batch(){
    std::vector<tensor>images;
    std::vector<int>labels;

    images.reserve(this->batch_size);
    labels.reserve(this->batch_size);

    for(int i=0;i<this->batch_size;++i){
        auto sample=this->add_to_buffer(i);
        images.emplace_back(sample.first);
        labels.emplace_back(sample.second);
    }

    return std::make_pair(std::move(images),std::move(labels));
}
//将图片放入缓冲区
std::pair<tensor,int> DataLoader::add_to_buffer(const int batch_index){
    ++this->iter;
    if(this->iter>=this->images_num){
        this->iter=0;
        if(this->shuffle)
            std::shuffle(this->images_list.begin(), this->images_list.end(), std::default_random_engine(this->seed));          
    }

    const auto& image_path=this->images_list[this->iter].first;
    const int image_label=this->images_list[this->iter].second;

    cv::Mat origin= cv::imread(image_path, cv::IMREAD_COLOR);

    //对图片进行增强
    if(this->augment)this->augmentor.make_augmentations(origin);
    //resize
    cv::resize(origin,origin,cv::Size(this->W,this->H));
    this->buffer[batch_index]->read_from_opencv_mat(origin.data);

    return std::make_pair(this->buffer[batch_index],image_label);
} 