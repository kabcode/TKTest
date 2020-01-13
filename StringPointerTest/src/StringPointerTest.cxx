#include <string>

auto Foo(int i) ->void
{
    std::string moreInformation;

    if (i % 1000 == 0)
        moreInformation = "Hello!";
}

auto Bar(int i) -> void
{
    std::string* moreInformation = nullptr;

    if (i % 1000 == 0)
    {
        moreInformation = new std::string();
        *moreInformation = "Hello!";
    }
        

    delete moreInformation;
}

int main()
{

    for (auto i = 0; i < 1000000; ++i)
    {
        Foo(i);
        Bar(i);
    }

    return EXIT_SUCCESS;
}