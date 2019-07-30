
#include "Windows.h"
#include <string>
#include <vector>
#include <iostream>
#include <thread>
#include <mutex>
#include <tchar.h>
#include <atomic>
#include <queue>
#include <filesystem>
#include <future>

namespace fs = std::experimental::filesystem;

void RefreshDirectory(LPTSTR, std::atomic_int& number);
void WatchDirectory(LPTSTR);

std::mutex FilenameMutex;
HANDLE hStopEvent;

DWORD processDirectoryChanges(const char *buffer, std::string& inputfilename);

void MonitorDirectory(std::string directory);

int main(int argc, char* argv[])
{

	if (argc < 3)
		return EXIT_FAILURE;

	std::vector<std::thread> threads(0);
	hStopEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

	for( auto i = 0; i < argc-1; ++i)
	{
		auto th = std::thread(WatchDirectory, argv[i+1]);
		threads.push_back(std::move(th));
	}

	std::this_thread::sleep_for(std::chrono::milliseconds(15000));
	SetEvent(hStopEvent);

	for (auto i = 0; i < threads.size(); ++i)
	{
		threads[i].join();
	}
		
	return 0;
}

DWORD processDirectoryChanges(const char *buffer, std::string& inputfilename)
{
	DWORD offset = 0;
	char fileName[MAX_PATH] = "";
	FILE_NOTIFY_INFORMATION *fni = NULL;

	do
	{
		fni = (FILE_NOTIFY_INFORMATION*)(&buffer[offset]);
		// since we do not use UNICODE, 
		// we must convert fni->FileName from UNICODE to multibyte
		int ret = ::WideCharToMultiByte(CP_ACP, 0, fni->FileName,
			fni->FileNameLength / sizeof(WCHAR),
			fileName, sizeof(fileName), NULL, NULL);

		switch (fni->Action)
		{
		case FILE_ACTION_ADDED:
		{
			std::lock_guard<std::mutex> guard(FilenameMutex);
			inputfilename = fileName;
		}
		break;
		default:
			break;
		}

		::memset(fileName, '\0', sizeof(fileName));
		offset += fni->NextEntryOffset;

	} while (fni->NextEntryOffset != 0);

	return 0;
}

void MonitorDirectory(std::string directory)
{
	auto stemp = std::wstring(directory.begin(), directory.end());
	LPCWSTR sw = stemp.c_str();

	auto directoryHandle = ::CreateFileW(sw,
		FILE_LIST_DIRECTORY,
		FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
		NULL,
		OPEN_EXISTING,
		(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED),
		NULL);


	const unsigned long dwFlags = FILE_NOTIFY_CHANGE_FILE_NAME |
		FILE_NOTIFY_CHANGE_DIR_NAME |
		FILE_NOTIFY_CHANGE_ATTRIBUTES |
		FILE_NOTIFY_CHANGE_SIZE |
		FILE_NOTIFY_CHANGE_LAST_WRITE;
	DWORD dwBytes = 0;

	OVERLAPPED overlapped = { 0 };
	overlapped.hEvent = ::CreateEventW(NULL, TRUE, FALSE, NULL);
	if (NULL == overlapped.hEvent)
		return;
		//return ::GetLastError();

	DWORD error = 0;
	char buffer[1024] = {};
	std::string filename("");

	if (!::ReadDirectoryChangesW(directoryHandle, buffer, sizeof(buffer), FALSE, dwFlags, &dwBytes, &overlapped, NULL))
	{
		error = GetLastError();
		if (error != ERROR_IO_PENDING)
			return;
	}

	while (true)
	{

		BOOL wait = FALSE;
		DWORD dw;

		BOOL Success = ::GetOverlappedResult(directoryHandle, &overlapped, &dw, wait);
		if (!Success)
		{
			error = GetLastError();
			if (error == ERROR_IO_INCOMPLETE) continue;
			return;
		}

		processDirectoryChanges(buffer, filename);
		std::cout << "Monitor: "<< filename << std::endl;
		::Sleep(1000);

		::ResetEvent(overlapped.hEvent);

		if (!::ReadDirectoryChangesW(directoryHandle, buffer, sizeof(buffer), FALSE, dwFlags, &dwBytes, &overlapped, NULL))
		{
			error = GetLastError();
			if (error != ERROR_IO_PENDING)
				return;
		}
	}

}

void WatchDirectory(LPTSTR lpDir)
{
	DWORD dwWaitStatus;
	HANDLE dwChangeHandles[2];
	TCHAR lpDrive[4];
	TCHAR lpFile[_MAX_FNAME];
	TCHAR lpExt[_MAX_EXT];

	_tsplitpath_s(lpDir, lpDrive, 4, NULL, 0, lpFile, _MAX_FNAME, lpExt, _MAX_EXT);

	lpDrive[2] = (TCHAR)'\\';
	lpDrive[3] = (TCHAR)'\0';

	// Watch the directory for file creation and deletion. 
	dwChangeHandles[0] = FindFirstChangeNotification(
		lpDir,                         // directory to watch 
		FALSE,                         // do not watch subtree 
		FILE_NOTIFY_CHANGE_FILE_NAME); // watch file name changes 

	if (dwChangeHandles[0] == INVALID_HANDLE_VALUE)
	{
		printf("\n ERROR: FindFirstChangeNotification function failed.\n");
		ExitProcess(GetLastError());
	}

	// Make a final validation check on our handles.
	if (dwChangeHandles[0] == NULL)
	{
		printf("\n ERROR: Unexpected NULL from FindFirstChangeNotification.\n");
		ExitProcess(GetLastError());
	}

	dwChangeHandles[1] = hStopEvent;
	if (dwChangeHandles[1] == INVALID_HANDLE_VALUE)
	{
		printf("\n ERROR: FindFirstChangeNotification function failed.\n");
		ExitProcess(GetLastError());
	}
	if (dwChangeHandles[1] == NULL)
	{
		printf("\n ERROR: Unexpected NULL from FindFirstChangeNotification.\n");
		ExitProcess(GetLastError());
	}

	// Change notification is set. Now wait on the notification 
	// handle and refresh accordingly.
	std::string filename("");
	DWORD error = 0;
	char buffer[1024] = {};
	const unsigned long dwFlags = FILE_NOTIFY_CHANGE_FILE_NAME |
		FILE_NOTIFY_CHANGE_DIR_NAME |
		FILE_NOTIFY_CHANGE_ATTRIBUTES |
		FILE_NOTIFY_CHANGE_SIZE |
		FILE_NOTIFY_CHANGE_LAST_WRITE;
	DWORD dwBytes = 0;
	OVERLAPPED overlapped = { 0 };
	overlapped.hEvent = ::CreateEventW(NULL, TRUE, FALSE, NULL);
	if (NULL == overlapped.hEvent)
		return; //return ::GetLastError();
	BOOL wait = TRUE;
	DWORD dw;

	BOOL Success = FALSE;

	auto directoryHandle = ::CreateFile(_T(lpDir),
		FILE_LIST_DIRECTORY,
		FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
		NULL,
		OPEN_EXISTING,
		(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED),
		NULL);
	
	std::atomic_int number(0);
	std::queue<std::string> Filenames;
	auto stop = false;

	while (!stop)
	{
		// Wait for notification.
		dwWaitStatus = ::WaitForMultipleObjects(2, dwChangeHandles, FALSE, INFINITE);

		switch (dwWaitStatus)
		{
		case WAIT_OBJECT_0:

			// A file was created, renamed, or deleted in the directory.
			// Refresh this directory and restart the notification.
			// A directory was created, renamed, or deleted.
			// Refresh the tree and restart the notification.
			if (!::ReadDirectoryChangesW(directoryHandle, buffer, sizeof(buffer), FALSE, dwFlags, &dwBytes, &overlapped, NULL))
			{
				error = GetLastError();
				if (error != ERROR_IO_PENDING)
					return;
			}

			Success = ::GetOverlappedResult(directoryHandle, &overlapped, &dw, wait);
			if (!Success)
			{
				error = GetLastError();
				if (error == ERROR_IO_INCOMPLETE)
					return;
			}

			processDirectoryChanges(buffer, filename);
			++number;
			std::cout << lpDir << ": " << filename << std::endl;
			Filenames.push(filename);
			::ResetEvent(overlapped.hEvent);
			if (FindNextChangeNotification(dwChangeHandles[0]) == FALSE)
			{
				printf("\n ERROR: FindNextChangeNotification function failed.\n");
				ExitProcess(GetLastError());
			}
			break;

		case WAIT_TIMEOUT:

			// A timeout occurred, this would happen if some value other 
			// than INFINITE is used in the Wait call and no changes occur.
			// In a single-threaded environment you might not want an
			// INFINITE wait.

			printf("\nNo changes in the timeout period.\n");
			break;

		case WAIT_OBJECT_0 +1:
			printf("\nNo changes in the timeout period.\n");
			stop = true;
			break;

		default:
			printf("\n ERROR: Unhandled dwWaitStatus.\n");
			ExitProcess(GetLastError());
		}
	}
}

void RefreshDirectory(LPTSTR lpDir, std::atomic_int& number)
{
	// This is where you might place code to refresh your
	// directory listing, but not the subtree because it
	// would not be necessary.
	++number;
	//_tprintf(TEXT("Directory (%s) changed.\n"), lpDir);
}