
#include <mutex>
#include <chrono>
#include <thread>
#include <iostream>

class locker
{
	std::vector<std::thread> threads;
	std::vector<int> number{0,0,0};
	std::vector<std::unique_lock<std::mutex>> vloc;
	std::vector<std::mutex> mut{3};

public:
	locker()
	{
		InitializeLocks();
	}

	void RunThreads(int n)
	{
		for (int i = 0; i < n; ++i)
		{
			threads.push_back(std::thread{ [=] { InsideThread(i); } });
		}
			
	}

	void InsideThread(int n)
	{
		try
		{
			vloc[n].lock();
			std::cout << std::this_thread::get_id() << std::endl;
			this->number[n] = this->number[n] + n;
			std::this_thread::sleep_for(std::chrono::seconds(1));
			vloc[n].unlock();
		}
		catch (const std::exception& e)
		{
			e.what();
		}
		
	}

	void JoinThreads()
	{
		for (auto i = 0; i < threads.size(); ++i)
		{
			threads[i].join();
		}
		for (int i = 0; i < number.size(); ++i)
		{
			std::cout << number[i] << std::endl;
		}	
	}

	void InitializeLocks()
	{
		for (auto& m : mut)
		{
			vloc.push_back(std::unique_lock<std::mutex>(m));
		}
		for (auto& v : vloc)
		{
			std::cout << "Locked!" << std::endl;
			v.unlock();
		}
	}
};

int main()
{
	
	auto loc = new locker();
	loc->RunThreads(3);
	loc->JoinThreads();
	delete loc;


	return 0;
}


