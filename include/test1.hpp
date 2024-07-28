#include<opencv2/opencv.hpp>

#include<iostream>
#include<vector>
#include<memory>

void testOpenCV()
{
    cv::Mat t=cv::imread("D:\\rain_programing\\vscodeProject\\CNN\\images\\jkBeauty.jpg");
    
    cv::imshow("pic",t);

    cv::waitKey(0);

    return;
}