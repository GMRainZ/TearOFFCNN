#include"data_format.hpp"
#include<string>
#include<iomanip>
#include<iostream>


void Tensor3D::read_from_opencv_mat(const uchar* const img_ptr)
{
    const int length=H*W;
    const int length_2=2*length;

    for(int i=0;i<length_2;i++)
    {
        const int p=3*i;
        data[i]=img_ptr[p]*1.f/255;
        data[length+i]=img_ptr[p+1]*1.f/255;
        data[length+i]=img_ptr[p+2]*1.f/255;
    }
}

void Tensor3D::set_zero()
{
    const int length=C*H*W;
    std::memset(data,0,sizeof(data_type)*length);
}

data_type Tensor3D::max()const
{
    return this->data[argmax()];
}

int Tensor3D::argmax()const
{
    const int length=C*H*W;
    if(!data)return 0;


    int max_index=0;
    data_type max_value=data[0];
    for(int i=1;i<length;i++)
    {
        if(data[i]>max_value)
        {
            max_value=data[i];
            max_index=i;
        }
    }
    return max_index;
}


data_type Tensor3D::min() const
{
    return this->data[argmin()];
}

int Tensor3D::argmin() const
{
    const int length=C*H*W;
    if(!data)return 0;

    int min_index=0;
    data_type min_value=data[0];
    for(int i=1;i<length;i++)
    {
        if(data[i]<min_value)
        {
            min_value=data[i];
            min_index=i;
        }
    }
    return min_index;
}


void Tensor3D::div(const data_type& times)
{
    const int length=C*H*W;
    for(int i=0;i<length;i++)
    {
        data[i]/=times;
    }
}

void Tensor3D::normalize(const std::vector<data_type> mean, const std::vector<data_type> std_div)
{
    if(C!=3)return;

    const int ch_size=H*W;
    for(int ch=0;ch<C;ch++)
    {
        data_type* const ch_ptr=data+ch*ch_size;
        for(int i=0;i<ch_size;i++)
            ch_ptr[i]=(ch_ptr[i]-mean[ch])/std_div[ch];
    }
}

cv::Mat Tensor3D::opencv_mat(const int CH)const
{
    cv::Mat origin;
    if(CH==3)
    {
        origin=cv::Mat(H,W,CV_8UC3);
        const int length=H*W;
        for(int i=0;i<length;i++)
        {
            const int p=3*i;
            origin.data[p]=cv::saturate_cast<uchar>(255 * data[i]);
            origin.data[p+1]=cv::saturate_cast<uchar>(255 * data[length+i]);
            origin.data[p+2]=cv::saturate_cast<uchar>(255 * data[length+i]);
        }
    }
    else if(C==1)
    {
        origin=cv::Mat(H,W,CV_8UC1);
        const int length=H*W;
        for(int i=0;i<length;i++)
            origin.data[i]=cv::saturate_cast<uchar>(255 * data[i]);
    }

    return origin;
}

int Tensor3D::get_length()const
{
    return C*H*W;
}

void Tensor3D::print_shape()const
{
    std::cout << this->name << "  ==>  " << this->C << " x " << this->H << " x " << this->W << "\n";
}


void Tensor3D::print(const int _C)const
{
    
    std::cout << this->name << "  content is : ";

    const int start=_C*H*W;
    for(int i=0;i<H;i++)
    {
        for(int j=0;j<W;j++)
        {
            std::cout << std::fixed << std::setprecision(3) << data[start+i*W+j] << "  ";
        }
        std::cout << "\n";
    }
}



std::shared_ptr<Tensor3D> Tensor3D::rot180()const
{
    std::shared_ptr<Tensor3D> res=std::make_shared<Tensor3D>(C,H,W,this->name+"_rot180");
    const int ch_size=H*W;
    for(int ch=0;ch<C;ch++)
    {
        const data_type* const ch_ptr=data+ch*ch_size;
        data_type* const ch_res_ptr=res->data+ch*ch_size;
        for(int i=0;i<ch_size;i++)
            ch_res_ptr[i]=ch_ptr[ch_size-1-i];
    }
    
    return res;
}


std::shared_ptr<Tensor3D> Tensor3D::pad(const int padding)const
{
    
    std::shared_ptr<Tensor3D> padded=std::make_shared<Tensor3D>(C,H+2*padding,W+2*padding,this->name+"_pad");

    const int new_W=W+2*padding;
    const int new_H=H+2*padding;
    const int new_ch_size=new_H*new_W;
    const int ch_size=H*W;

    std::memset(padded->data,0,sizeof(data_type)*new_ch_size*C);

    
    for(int ch=0;ch<C;ch++)
    {
        for(int i=0;i<H;i++)
        {
            std::memcpy(padded->data+ch*new_ch_size+(i+padding)*new_W+padding,
                data+ch*ch_size+i*W,W*sizeof(data_type));
        }
    }

    return padded;
}

Tensor3D::~Tensor3D()noexcept
{
    if(this->data)
    {
        delete[] this->data;
        this->data=nullptr;
    }
}