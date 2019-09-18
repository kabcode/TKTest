#include "obj.h"
#include <thread>
#include <iostream>
#include <future>

void obj::startThread()
{
    int msg(1);
    std::thread t1(&obj::task1, this, msg);
    t1.join();

    auto fut = std::async(&obj::returnInt,this, 2);
    auto res = fut.get();

    std::cout << "async says: " << res << std::endl;
}

void obj::task1(int i)
{
    std::cout << "task1 says: " << i;
}

int obj::returnInt(int i)
{
    return i * 2;
}

