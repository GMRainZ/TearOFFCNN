#include"test1.hpp"



int main()
{
    printf("Hello world!\n");

    std::cout<<"nihao shijie\n"<<std::endl;
    std::vector<int>iv(10,8);

    std::shared_ptr<std::vector<int>>isp(std::make_shared<std::vector<int>>(
        10,8
    ));

    std::cout<<( *isp==iv )<<std::endl;

    testOpenCV();

    return 0;
}