
#include <mutex>
#include <chrono>
#include <thread>
#include <iostream>

class locker
{
	std::vector<std::thread> threads;
	std::vector<int> number{ 0,0,0 };
	std::vector<std::unique_lock<std::mutex>> vloc;
	std::mutex mut;
	std::mutex* mptr;

public:
	locker()
	{
		//InitializeLocks();
		mptr = &mut;
	}

	void RunThreads(int n)
	{
		for (int i = 0; i < n; ++i)
		{
			threads.push_back(std::thread{ [=] { InsideThread(i); } });
		}
	}

	void RunAgain(int n)
	{

		for (int i = 0; i < n; ++i)
		{
			threads.push_back(std::thread{ [=] { OutsideThread(i); } });
		}

	}

	void InsideThread(int n)
	{
		try
		{
			mptr->lock();
			std::cout << std::this_thread::get_id() << std::endl;
			this->number[n] = this->number[n] + n;
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}
		catch (const std::exception& e)
		{
			e.what();
		}
	}

	void OutsideThread(int n)
	{
		try
		{
			this->number[n] = this->number[n] + n;
			std::this_thread::sleep_for(std::chrono::seconds(1));
			mptr->unlock();
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
		threads.clear();
	}

	void InitializeLocks()
	{
		/*for (auto& m : mut)
		{
			vloc.push_back(std::unique_lock<std::mutex>(m));
		}
		for (auto& v : vloc)
		{
			std::cout << "Locked!" << std::endl;
			v.unlock();
		}*/
	}
};

int main()
{

	auto loc = new locker();
	loc->RunThreads(3);
	loc->JoinThreads();
	loc->RunAgain(3);
	loc->JoinThreads();
	delete loc;


	return 0;
}


