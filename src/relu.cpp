
#include"architectures.hpp"

using namespace architectures;

std::vector<tensor> ReLU::forward(const std::vector<tensor>&input){ 
    //获取图像的信息
    const int batch_size = input.size();

    //第一次经过这一层
    if(output.empty()){
        //给输出分配空间
        this->output.reserve(batch_size);
        for(int i = 0; i < batch_size; i++){
            this->output.emplace_back(new Tensor3D(input[i]->C, input[i]->H, input[i]->W,this->name+"_output_"+std::to_string(i)));
        }
    }
    const int total_length=input[0]->get_length();
    for(int i = 0; i < batch_size; i++){
        data_type* const src_ptr = input[i]->data;
        data_type* const dst_ptr = this->output[i]->data;
        for(int j = 0; j < total_length; j++){
            this->output[i]->data[j] = input[i]->data[j] > 0 ? input[i]->data[j] : 0;
        }
    }

    return this->output;
}

std::vector<tensor> ReLU::backward(const std::vector<tensor>&delta){
    const int batch_size = delta.size();
    const int total_length = delta[0]->get_length();

    for(int i = 0; i < batch_size; i++){
        data_type* const src_ptr = delta[i]->data;
        data_type* const dst_ptr = delta[i]->data;
        for(int j = 0; j < total_length; j++){
            delta[i]->data[j] = delta[i]->data[j] > 0 ? src_ptr[j] : 0;
        }
    }

    for(int i = 0; i < batch_size; i++){
        delta[i]->name = this->name+"_delta_"+std::to_string(i);
    }

    return delta;
}


//end of file