#include"architectures.hpp"

#include<vector>
#include<random>

using namespace architectures;

LinearLayer::LinearLayer(std::string _name,const int _in_channels,const int _out_channels)
    : Layer(_name),in_channels(_in_channels),out_channels(_out_channels)
{
    // Initialize weights and biases
    std::default_random_engine e(1998);
    std::normal_distribution<double> engine(0.0,1.0);
    for(int i=0;i<out_channels;++i){
        bias.emplace_back(engine(e))/random_times;
    }
    const int length = in_channels*out_channels;
    for(int i=0;i<length;++i){
        weights.emplace_back(engine(e))/random_times;
    }
}


std::vector<tensor>LinearLayer::forward(const std::vector<tensor>& input){
    //获取输入的信息
    const int batch_size=input.size();
    this->delta_shape=input[0]->get_shape();

    //清空之前的结果，重新开始
    std::vector<tensor>().swap(output);

    for(int i=0;i<batch_size;++i){
        output.emplace_back(new Tensor3D(out_channels,this->name+"_output"+std::to_string(i)));
    }
    
    //记录输入
    if(!no_grad)this->__input=input;

    //batch每个图像分开计算
    for(int b=0;b<batch_size;++b){
        //计算输出
        data_type*src_ptr=input[b]->data;
        data_type*dst_ptr=output[b]->data;

        for(int i=0;i<out_channels;++i){
            dst_ptr[i]=bias[i];
            for(int j=0;j<in_channels;++j){
                dst_ptr[i]+=src_ptr[j]*weights[j*in_channels+i];
            }
        }
    }
    return output;
}

std::vector<tensor>LinearLayer::backward(std::vector<tensor>& delta){
    //获取delta的信息
    const int batch_size=delta.size();

    //第一次回传，给缓冲区的梯度分配空间
    if(this->weights_gradients.empty()){
        this->weights_gradients.resize(in_channels*out_channels);
        this->bias_gradients.resize(out_channels);
    }

    //计算W的梯度
    for(int i=0;i<in_channels;++i){
        data_type*w_ptr=this->weights_gradients.data()+i*out_channels;
        for(int j=0;j<out_channels;++j){
            w_ptr[j]=0;
            for(int b=0;b<batch_size;++b){
                w_ptr[j]+=delta[b]->data[j]*this->__input[b]->data[i];
            }
            w_ptr[j]/=batch_size;
        }
    }

    //更新bias
    for(int i=0;i<out_channels;++i){
        data_type sum_value=0;
        for(int b=0;b<batch_size;++b){
            sum_value+=delta[b]->data[i];   
        }
        this->bias_gradients[i]=sum_value/batch_size;
    }


    //如果是第一次回传
    if(this->delta_output.empty()){
        this->delta_output.reserve(batch_size);
        for(int i=0;i<batch_size;++i)
            this->delta_output.emplace_back(new Tensor3D(delta_shape,this->name+"_delta_output"+std::to_string(i)));
    }

    //计算返回的梯度，大小和__input一样
    for(int b=0;b<batch_size;++b){
        data_type*src_ptr=delta[b]->data;
        data_type*dst_ptr=this->delta_output[b]->data;
        for(int i=0;i<in_channels;++i){
            dst_ptr[i]=0;
            data_type*w_ptr=this->weights.data()+i*out_channels;
            for(int j=0;j<out_channels;++j){
                dst_ptr[i]+=src_ptr[j]*w_ptr[j];
            }
        }
    }

    return this->delta_output;
}

void LinearLayer::update_gradients(const data_type learning_rate){
    //判断是否为空
    assert(!this->weights_gradients.empty());

    //更新权重和偏置
    const int total_length=in_channels*out_channels+out_channels;
    for(int i=0;i<total_length;++i)
        this->weights[i]-=learning_rate*this->weights_gradients[i];
    for(int i=0;i<out_channels;++i)
        this->bias[i]-=learning_rate*this->bias_gradients[i];
}


void LinearLayer::save_weights(std::ofstream&writer)const{
    writer.write(reinterpret_cast<const char*>(&weights[0]),static_cast<int>(sizeof(data_type)*in_channels*out_channels));

    writer.write(reinterpret_cast<const char*>(&bias[0]),static_cast<int>(sizeof(data_type)*out_channels));
}

void LinearLayer::load_weights(std::ifstream&reader){
    reader.read(reinterpret_cast<char*>(&weights[0]),static_cast<std::streamsize>(sizeof(data_type)*in_channels*out_channels));

    reader.read(reinterpret_cast<char*>(&bias[0]),static_cast<std::streamsize>(sizeof(data_type)*out_channels));
}