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
        virtual void save_weights(std::ofstream& writer)const{};
        virtual void load_weights(std::ifstream& reader){};
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
        Conv2D(const std::string _name,const int _in_channels=3,
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
        virtual void save_weights(std::ofstream& writer)const;
        
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
        int get_params_num()const;
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
        std::vector<tensor>delta_output;

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
        const int out_channels;
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
         * @brief 保存回传信息
         * 
         */
        std::vector<tensor>__input;

        /**
         * @brief delta回传到输入的梯度
         * 
         */
        std::vector<tensor>delta_output;

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
        LinearLayer(std::string _name,const int _in_channels,const int _out_channels);   

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
        std::vector<tensor>backward(const std::vector<tensor>&delta);

        virtual void update_gradients(const data_type learning_rate=1e-4);
        virtual void save_weights(std::ofstream& writer)const;
        virtual void load_weights(std::ifstream& reader);
    };



    class BatchNorm2D : public Layer{
    private:
        const int out_channels;
        const data_type eps;
        const data_type momentum;

        //要学习的参数
        std::vector<data_type>gamma;
        std::vector<data_type>beta;

        //history information
        std::vector<data_type>moving_mean;
        std::vector<data_type>moving_var;

        //缓冲区
        std::vector<tensor>normed_input;;
        std::vector<data_type>buffer_mean;
        std::vector<data_type>buffer_var;

        //保留梯度信息
        std::vector<data_type>gamma_gradients;
        std::vector<data_type>beta_gradients;

        //临时梯度
        tensor norm_graients;

        //求梯度时用得到
        std::vector<tensor>__input;

    public:
        /**
         * @brief Construct a new BatchNorm2D object
         * 
         */
        BatchNorm2D(std::string _name,const int _out_channels,const data_type _eps=1e-5,const data_type _momentum=0.1);

        /**
         * @brief forward  
         * 
         */
        std::vector<tensor>forward(const std::vector<tensor>&input);

        /**
         * @brief backward
         * 
         */
        std::vector<tensor>backward(const std::vector<tensor>&delta);
    
        /**
         * @brief 更新学习率
         * 
         * @param learning_rate 
         */
        virtual void update_gradients(const data_type learning_rate=1e-4);

        /**
         * @brief 保存权值
         * 
         * @param writer 
         */
        virtual void save_weights(std::ofstream& writer)const;
        
        /**
         * @brief 读取权值
         * 
         * @param reader 
         */
        virtual void load_weights(std::ifstream& reader);
    };


    class Dropout : public Layer{
    private:
        const data_type p;
        int selected_num;
        std::vector<int>sequence;
        std::default_random_engine drop;
        std::vector<int>mask;
    public:
        /**
         * @brief Construct a new Dropout object
         * 
         * @param _name 
         * @param _p 
         */
        Dropout(std::string _name,const data_type _p=0.5)
            :Layer(_name),p(_p),drop(std::chrono::system_clock::now().time_since_epoch().count()){}
        
        /**
         * @brief forward
         * 
         * @param input 
         */
        std::vector<tensor>forward(const std::vector<tensor>&input);
        
        /**
         * @brief backward
         * 
         * @param delta 
         */
        std::vector<tensor>backward(const std::vector<tensor>&delta);
    };

    class AlexNet{
    public:
        /**
         * @brief 是否打印信息
         * 
         */
        bool print_info=false;
    private:
        /**
         * @brief 各层
         * 
         */
        std::list<std::shared_ptr<Layer>>layers_sequence;
    public:
        /**
         * @brief Construct a new Alex Net object
         * 
         * @param num_classes 
         * @param batch_norm 
         */
        AlexNet(const int num_classes=10,const bool batch_norm=false);

        /**
         * @brief forward
         * 
         * @param input 
         */
        std::vector<tensor>forward(const std::vector<tensor>&input);
        /** 
         * @brief backward
         */
        void backward(std::vector<tensor>&delta_start);

        /**
         * @brief 更新权值
         * 
         * @param learning_rate 
         */
        void update_gradients(const data_type learning_rate=1e-4);

        /**
         * @brief 保存权值
         * 
         * @param writer 
         */
                // 保存模型
        void save_weights(const std::filesystem::path& save_path) const;
    
        /**
         * @brief 加载权值
         * 
         * @param reader 
         */

        // 加载模型
        void load_weights(const std::filesystem::path& checkpoint_path);

        cv::Mat grad_cam(const std::string&layer_name)const;
    };
}



#endif // CNN_ARCHITECTURES_HPP