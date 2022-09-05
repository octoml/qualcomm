#include <iostream>

#include "implementations/conv2d_vgg16.h"

int main()
{
    std::string res = measureExecTime(conv2d_vgg16, 10);
    std::cout << "OpenCLRunner, Exec time: " << res << std::endl;
    return 0;
}
