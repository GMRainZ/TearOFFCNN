#include"architectures.hpp"

using namespace architectures;


std::vector<tensor>Dropout::forward(const std::vector<tensor>&input){
    const int batch_size = input.size();
    const int out_channels = input[0]->C;
    const int area= input[0]->H*input[0]->W;

    //第一次经过，分配空间
    if(this->sequence.empty()){
        this->sequence.assign(out_channels,0); //初始化为0
        for(int o=0;o<out_channels;o++){
            this->sequence[o] = o;
        }
        this->selected_num=int(p*out_channels);

        assert(out_channels>selected_num);

        this->mask.assign(out_channels,0);

        this->output.reserve(batch_size);

        for(int b=0;b<batch_size;b++){
            this->output.emplace_back(new Tensor3D(out_channels,input[b]->H,input[b]->W));
        }
    }


    //随机选择
    std::shuffle(this->sequence.begin(),this->sequence.end(),this->drop);

    //如果是训练阶段
    if(!no_grad){
        //记录被选中的卷积核（输出通道），前selected_num个失活，其余置为-1
        for(int i=0;i<out_channels;i++){
            this->mask[i]=i>=this->selected_num?this->sequence[i]:-1;
        }
        const auto copy_size=sizeof(data_type)*area;
        for(int b=0;b<batch_size;b++){
            for(int o=0;o<out_channels;o++){
                if(o>=this->selected_num){
                    memcpy(this->output[b]->data+o*area,input[b]->data+o*area,copy_size);
                }else{
                    memset(this->output[b]->data+o*area,0,copy_size);
                }
            }
        }
    }else{//验证或者测试阶段
        //直接将结果1-p,选中1-p的输出
        const auto length=input[0]->get_length();
        const data_type prob=1-this->p;

        for(int b=0;b<batch_size;b++){
            data_type*const src_ptr=input[b]->data;
            data_type*const dst_ptr=this->output[b]->data;
            for(int i=0;i<length;i++){
                dst_ptr[i]=src_ptr[i]*prob;
            }
        }
    }

    return this->output;
}




std::vector<tensor>Dropout::backward(const std::vector<tensor>&delta){
    const int batch_size = delta.size();
    const int out_channels = delta[0]->C;
    const int area= delta[0]->H*delta[0]->W;

    //根据mask来操作
    for(int b=0;b<batch_size;b++){
        for(int o=0;o<out_channels;o++){
            if(this->mask[o]==-1){//这个卷积核被失活
                std::memset(delta[b]->data+o*area,0,area*sizeof(data_type));
            }
        }
    }

    return delta;
}