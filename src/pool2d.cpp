#include"architectures.hpp"

using namespace architectures;


std::vector<tensor> MaxPool2D::forward(const std::vector<tensor>&input){
    /**
     * @brief 获取信息
     * 
     */
    const int batch_size = input.size();
    const int H=input[0]->H,W=input[0]->W,C=input[0]->C;
    
    /**
     * @brief 输出图片的大小
     * 
     */
    const int out_H = floor(((H - kernel_size + 2 * padding) / step))+1;
    const int out_W = floor(((W - kernel_size + 2 * padding) / step))+1;

    //第一次经过池化层
    if(this->output.empty()){
        //分配空间
        // this->output = std::vector<tensor>(batch_size);
        this->output.reserve(batch_size);
        for(int i=0;i<batch_size;i++) 
            this->output.emplace_back(new Tensor3D(C,H,W,this->name+"_output_"+std::to_string(i)));

        //给delta分配空间
        if(!no_grad){
            this->delta_output.reserve(batch_size);
            for(int i=0;i<batch_size;i++) 
                this->delta_output.emplace_back(new Tensor3D(C,out_H,out_W,this->name+"_delta_output_"+std::to_string(i)));
            
            this->mask.reserve(batch_size);
            for(int i=0;i<batch_size;i++) 
                this->mask.emplace_back(std::vector<int>(C*out_H*out_W,0));
        }
        int pos=0;
        for(int i=0;i<kernel_size;i++)
            for(int j=0;j<kernel_size;j++)
                offset[pos++]=i*W+j;
    }
    // 如果存在 backward, 每次 forward 要记得把 mask 全部填充为 0
    const int out_length = out_H * out_W;
    int* mask_ptr = nullptr;
    if(!no_grad) {
        const int mask_size = C * out_length;
        for(int b = 0;b < batch_size; ++b) {
            int* const mask_ptr = this->mask[b].data();
            for(int i = 0;i < mask_size; ++i) mask_ptr[i] = 0;
        }
    }

    //开始池化

    const int length=H*W;
    const int H_kernel=H-kernel_size,W_kernel=W-kernel_size;
    const int window_range=kernel_size*kernel_size;

    for(int b=0;b<batch_size;++b){
        // 16 X 111 X 111 → 16 X 55 X 55
        for(int i=0;i<C;++i){
            // 现在我拿到了第 b 张图的第 i 个通道, 一个指向内容大小 55 X 55 的指针
            data_type*const cur_image_features = input[b]->data + i * length;
            // 第 b 个输出的第 i 个通道的, 同样是指向内容大小 55 X 55 的指针
            data_type*const output_ptr= this->output[b]->data + i * out_length;
            // 记录第 b 个输出, 记录有效点在 111 X 111 这个图上的位置, 一共有 55 X 55 个值
            if(!no_grad){
                mask_ptr = this->mask[b].data() + i * out_length;
            }
            int cur=0;// 当前池化输出的位置

            for(int x=0;x<=H_kernel;x+=step){
                // 获取这个通道图像的第 x 行指针
                data_type*const row_ptr=cur_image_features+x*W;
                for(int y=0;y<=W_kernel;y+=step){
                    //找到局部的 kernel_size*kernel_size 的最大值
                    data_type max_value=row_ptr[y];
                    int max_pos=0;
                    for(int k=0;k<window_range;++k){
                        data_type cur_value=row_ptr[offset[k]+y];
                        if(cur_value>max_value){
                            max_value=cur_value;
                            max_pos=offset[k];
                        }
                    }
                    //局部最大值填到相应的位置上
                    output_ptr[cur]=max_value;

                    if(!no_grad){
                        max_pos+=x*W+y;
                        mask_ptr[cur]=max_pos;
                    }
                    ++cur;
                }
            }
        }
    }

    return this->output;
}



std::vector<tensor>MaxPool2D::backward(const std::vector<tensor>& delta){
    //获取输入的梯度信息
    const int batch_size=delta.size();
    for(int b=0;b<batch_size;++b){
        this->delta_output[b]->set_zero();
    }
    const int total_length=delta.size();
    // batch 每张图像, 根据 mask 标记的位置, 把 delta 中的值填到 delta_output 中去
    for(int b=0;b<batch_size;++b){
        int *mask_ptr=this->mask[b].data();
        //获取delta 的第b张输出传回的梯度的起始地址
        data_type*const src_ptr=delta[b]->data;
        //获取返回到输入的梯度，第b张梯度的起始地址
        data_type*const res_ptr=this->delta_output[b]->data;

        for(int i=0;i<total_length;++i){
            res_ptr[mask_ptr[i]]=src_ptr[i];
        }
    }

    return this->delta_output;
}