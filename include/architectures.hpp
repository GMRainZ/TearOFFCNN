#ifndef CNN_ARCHITECTURES_HPP
#define CNN_ARCHITECTURES_HPP

#include <list>
#include <fstream>
#include "pipeline.hpp"


namespace architectures {
    using namespace pipeline;


    extern data_type random_times;
    //是否要backward
    extern bool no_grad;

    class WithouGrad final{
    public:
        explicit WithouGrad(){
            architectures::no_grad = true;
        }
        ~WithouGrad(){
            architectures::no_grad = false;
        }
    };

    class Layer{
    public:
        const std::string name;//这一层的名字
        std::vector<tensor>output;//输出的张量
    public:
        Layer(const std::string& name):name(std::move(name)){}

        virtual std::vector<tensor> forward(const std::vector<tensor>&input) = 0;
        virtual std::vector<tensor> backward(const std::vector<tensor>&delta) = 0;
        virtual void update_gradients(const data_type learning_rate=1e-4){};
        virtual void save_wights(std::ofstream& writer)const{};
        virtual void load_wights(std::ifstream& reader){};
        virtual std::vector<tensor> get_output()const{return this->output;}
    };

    class ReLU : public Layer{
    public:
        ReLU(const std::string& name):Layer(name){}
        std::vector<tensor> forward(const std::vector<tensor>&input) override;
        std::vector<tensor> backward(const std::vector<tensor>&delta) override;
    };

    class Conv2D : public Layer{
    private:
        //卷积层的固有信息
        std::vector<tensor>weights;//卷积核的权值参数, out_channels X in_channels X kernel_size X kernel_size
        std::vector<data_type>bias;//偏置
        const int in_channels;//要滤波的特征图的通道数
        const int out_channels;//这一层的卷积核的个数
        const int kernel_size;//卷积核的边长
        const int stride;//步长
        const int params_for_one_kernel;//一个卷积核的参数个数
        const int padding=0;//填充
        std::default_random_engine seed;//初始化的种子
        std::vector<int>offset;//卷积的偏移量

        //history information
        std::vector<tensor>__input;// 求梯度需要, 其实存储的是指针
        //buffer
        std::vector<tensor>delta_output;//存储 回传到上一层的梯度
        std::vector<tensor>weights_gradients;//权值的梯度
        std::vector<data_type>bias_gradients;//bias的梯度
    
    public:
        /**
         * @brief Construct a new Conv 2 D object
         * 
         * @param _name 卷积层的名字
         * @param _in_channels 输入的通道数
         * @param _out_channels 输出的通道数
         * @param _kernel_size 卷积核的尺寸
         * @param _stride 步长
         */
        Conv2D(std::string _name,const int _in_channels=3,
            const int _out_channels=16,const int _kernel_size=3,
            const int _stride=2);
        //卷积操作的forward过程，batch_num X in_channels X height X width
        /**
         * @brief 前向传播
         * 
         * @param input 输入向量
         * @return std::vector<tensor> 
         */
        std::vector<tensor> forward(const std::vector<tensor>&input) override;

        //把堆上的数据放在栈区，局部变量快
        /**
         * @brief 反向传播
         * 
         * @param delta 变化量
         * @return std::vector<tensor> 
         */
        std::vector<tensor> backward(const std::vector<tensor>&delta) override;
    public:
        /**
         * @brief 更新梯度
         * 
         * @param learning_rate 学习率 
         */
        void update_gradients(const data_type learning_rate=1e-4) override;

        /**
         * @brief 存储权值
         * 
         * @param path 
         */
        virtual void save_weights(const std::string path)const;
        
        /**
         * @brief 加载权值
         * 
         * @param reader 
         */
        void load_weights(std::ifstream&reader);

        /**
         * @brief Get the params num object
         * 
         */
        void get_params_num()const;
    };

    class MaxPool2D:public Layer{
    private:
        /**
         * @brief 核的尺寸
         * 
         */
        const int kernel_size;
        /**
         * @brief 步长
         * 
         */
        const int step;
        /**
         * @brief 填充
         * 
         */
        const int padding;
        
        /**
         * @brief 
         * // 记录哪些位置是有梯度回传的, 第 b 张图, 每张图一个 std::vector<int>
         */
        std::vector<std::vector<int>>mask;

        /**
         * @brief 返回的delta
         * 
         */
        std::vector<std::vector<int>>delta_output;

        /**
         * @brief 偏移量指针
         * 
         */
        std::vector<int>offset;
    public:
        MaxPool2D(std::string _name,const int _kernel_size=2,
        const int _step=2,const int _padding=0)
        :Layer(_name),kernel_size(_kernel_size),step(_step),
        padding(_padding){}
    

        std::vector<tensor> forward(const std::vector<tensor>&input) override;
        std::vector<tensor> backward(const std::vector<tensor>&delta) override;
    };

    class LinearLayer : public Layer{
    private:
        /**
         * @brief 输入神经元的个数
         * 
         */
        const int in_channels;
        /**
         * @brief 输出神经元的个数
         * 
         */
        const int out_channel;
        /**
         * @brief 权值矩阵
         * 
         */
        std::vector<data_type>weights;
        /**
         * @brief 偏置
         * 
         */
        std::vector<data_type>bias;

        //history information
        /**
         * @brief delta的形状
         * 
         */
        std::tuple<int,int,int>delta_shape;
    
        /**
         * @brief 权值矩阵的梯度
         * 
         */
        std::vector<data_type>weights_gradients;
        /**
         * @brief bias的梯度
         * 
         */
        std::vector<data_type>bias_gradients;
    
    public:
        /**
         * @brief Construct a new Linear Layer object
         * 
         * @param _name 
         * @param _in_channels 
         * @param _out_channel 
         */
        LinearLayer(std::string _name,const int _in_channels,const int _out_channel);   

        /**
         * @brief 前向传播
         * 
         * @param input 
         */
        std::vector<tensor>forward(const std::vector<tensor>&input);
    
        /**
         * @brief 反向传播
         * @param delta 
         */
        std::vector<tensor>backward(std::vector<tensor>&delta);

        
    };
}



#endif // CNN_ARCHITECTURES_HPP