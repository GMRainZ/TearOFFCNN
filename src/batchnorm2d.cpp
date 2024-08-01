#include<string>
#include "architectures.hpp"

using namespace architectures;

namespace {
    inline data_type square(const data_type& x) {
        return x * x;
    }
}

BatchNorm2D::BatchNorm2D(std::string _name,const int _out_channels,const data_type _eps,const data_type _momentum)
    :Layer(_name),out_channels(_out_channels),eps(_eps),momentum(_momentum),
    gamma(_out_channels,1.0), beta(_out_channels,0.0),
    moving_mean(_out_channels,0.0),moving_var(_out_channels,0.0),
    buffer_mean(_out_channels,0.0),buffer_var(_out_channels,0.0)
        {}


std::vector<tensor> BatchNorm2D::forward(const std::vector<tensor>& input) {
    //获取信息
    const int batch_size = input.size();
    const int H=input[0]->H;
    const int W=input[0]->W;
    
    //第一次经过forwar层，分配空间
    if(output.empty()) {
        this->output.reserve(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            this->output.emplace_back(new Tensor3D(out_channels,H,W,this->name+"_output"+std::to_string(i)));
        }
        this->normed_input.reserve(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            this->normed_input.emplace_back(new Tensor3D(out_channels,H,W,this->name+"_normed_input"+std::to_string(i)));        
        }
    }

    //记录当前batch的均值和方差
    if(!no_grad)this->__input=input;

    //开始归一化
    const int feature_map_length = H * W;//二维图像的feature map长度
    const int output_length=batch_size*feature_map_length;//输出长度

    for(int out_channel=0;out_channel<out_channels;++out_channel){
        //如果是训练
        if (!no_grad) {
            //均值u
            data_type u=0.0;
            for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
                data_type*const src_data = input[batch_index]->data + out_channel*feature_map_length;
                for (int i = 0; i < feature_map_length; ++i) {
                    u += src_data[i];
                }
            
            }
            //如果有backward记住均值
            u /= output_length;

            //方差sigma
            data_type sigma=0.0;
            for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
                data_type*const src_data = input[batch_index]->data + out_channel*feature_map_length;
                for (int i = 0; i < feature_map_length; ++i) {
                    sigma += square(src_data[i]-u); 
                }
            }
            sigma /= output_length;

            if(!no_grad){
                buffer_mean[out_channel]=u;
                buffer_var[out_channel]=sigma;
            }

            //对第out_channel作归一化
            const data_type var_inv=1./std::sqrt(sigma+eps);
            for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
                data_type*const src_data = input[batch_index]->data + out_channel*feature_map_length;
                data_type*const norm_data= normed_input[batch_index]->data + out_channel*feature_map_length;
                data_type*const dst_data = output[batch_index]->data + out_channel*feature_map_length;

                for(int i=0;i<feature_map_length;++i){
                    norm_data[i]=(src_data[i]-u)*var_inv;
                    dst_data[i]=norm_data[i]*gamma[out_channel]+beta[out_channel];
                }
            }

            moving_mean[out_channel]=moving_mean[out_channel]*momentum+u*(1-momentum);
            moving_var[out_channel]=moving_var[out_channel]*momentum+sigma*(1-momentum);
        }else{
            //如果是推理
            //对第out_channel作归一化
            const data_type u=moving_mean[out_channel];
            const data_type var_inv=1./std::sqrt(moving_var[out_channel]+eps);
            for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
                data_type*const src_data = input[batch_index]->data + out_channel*feature_map_length;
                data_type*const norm_data= normed_input[batch_index]->data + out_channel*feature_map_length;
                data_type*const dst_data = output[batch_index]->data + out_channel*feature_map_length;

                for(int i=0;i<feature_map_length;++i){
                    norm_data[i]=(src_data[i]-u)*var_inv;
                    dst_data[i]=norm_data[i]*gamma[out_channel]+beta[out_channel];
                }

            }
        }
    }


    return this->output;
}


//batch norm 的 delta 也可以就地修改
std::vector<tensor>BatchNorm2D::backward(const std::vector<tensor> &delta){
    const int batch_size=delta.size();
    const int feature_map_length=delta[0]->H*delta[0]->W;
    const int output_length=feature_map_length*batch_size;

    //第一次返回则分配空间
    if(gamma_gradients.empty()){
        gamma_gradients.assign(out_channels,0);
        beta_gradients.assign(out_channels,0);
        norm_graients = std::make_shared<Tensor3D>(batch_size,delta[0]->H,delta[0]->W);
    }

    //每次使用之前梯度清零
    for(int i=0;i<out_channels;++i){
        gamma_gradients[i]=0;
        beta_gradients[i]=0;
    }
    //从后往前推

    for(int out_channel=0;out_channel<out_channels;++out_channel){
        //清空u，var，norm的梯度
        norm_graients->set_zero();

        //首先是对beta和gamma求，还有norm的梯度
        for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
            data_type*const src_data = delta[batch_index]->data + out_channel*feature_map_length;
            data_type*const norm_data= normed_input[batch_index]->data + out_channel*feature_map_length;
            data_type*const norm_g_data = norm_graients->data + batch_index*feature_map_length;
            for(int i=0;i<feature_map_length;++i){
                norm_g_data[i]+=src_data[i]*gamma[out_channel];
                gamma_gradients[out_channel]+=norm_data[i]*src_data[i];
                beta_gradients[out_channel]+=src_data[i];
            }
        
        }
        //然后是对u和var求梯度
        data_type var_gradient=0;
        const data_type u=buffer_mean[out_channel];
        const data_type var_inv=1./std::sqrt(buffer_var[out_channel]+eps);
        const data_type var_inv_3=var_inv*var_inv*var_inv;

        for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
            data_type*const src_ptr=__input[batch_index]->data + out_channel*feature_map_length;
            data_type*const norm_g_data = norm_graients->data + batch_index*feature_map_length;
            for(int i=0;i<feature_map_length;++i){
                var_gradient+=norm_g_data[i]*(src_ptr[i]-u)*(-0.5)*var_inv_3;
            }
        }

        //接下来求均值u的均值
        data_type u_gradients=0;
        const data_type inv = var_gradient/output_length;

        for(int batch_index=0;batch_index<batch_size;++batch_index){
            data_type*const src_ptr=__input[batch_index]->data + out_channel*feature_map_length;
            data_type*const norm_g_data = norm_graients->data + batch_index*feature_map_length;
            for(int i=0;i<feature_map_length;++i){
                u_gradients+=norm_g_data[i]*(-var_inv)+inv*(-2)*(src_ptr[i]-u);
            }
        }

        //最后求delta,返回给输入层的梯度
        for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
            data_type*const src_ptr=__input[batch_index]->data + out_channel*feature_map_length;
            data_type*const norm_g_data = norm_graients->data + batch_index*feature_map_length;
            data_type*const delta_ptr=delta[batch_index]->data + out_channel*feature_map_length;
            for(int i=0;i<feature_map_length;++i){
                delta_ptr[i]=norm_g_data[i]*var_inv+inv*2*(src_ptr[i]-u)+var_gradient*(src_ptr[i]-u)+u_gradients/output_length;
            }
            
        }


    }
    return delta;
}












void BatchNorm2D::update_gradients(const data_type learning_rate) {
    for(int o = 0;o < out_channels; ++o) {
        gamma[o] -= learning_rate * gamma_gradients[o];
        beta[o] -= learning_rate * beta_gradients[o];
    }
}

void BatchNorm2D::save_weights(std::ofstream& writer) const {
    const int stream_size = sizeof(data_type) * out_channels;
    writer.write(reinterpret_cast<const char *>(&gamma[0]), static_cast<std::streamsize>(stream_size));
    writer.write(reinterpret_cast<const char *>(&beta[0]), static_cast<std::streamsize>(stream_size));
    writer.write(reinterpret_cast<const char *>(&moving_mean[0]), static_cast<std::streamsize>(stream_size));
    writer.write(reinterpret_cast<const char *>(&moving_var[0]), static_cast<std::streamsize>(stream_size));
}

void BatchNorm2D::load_weights(std::ifstream& reader) {
    const int stream_size = sizeof(data_type) * out_channels;
    reader.read((char*)(&gamma[0]), static_cast<std::streamsize>(stream_size));
    reader.read((char*)(&beta[0]), static_cast<std::streamsize>(stream_size));
    reader.read((char*)(&moving_mean[0]), static_cast<std::streamsize>(stream_size));
    reader.read((char*)(&moving_var[0]), static_cast<std::streamsize>(stream_size));
}
