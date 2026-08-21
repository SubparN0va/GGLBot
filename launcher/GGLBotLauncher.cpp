#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#else
#include <cerrno>
#include <cstring>
#include <unistd.h>
#endif

namespace
{
    constexpr int TORCH_SEARCH_DEPTH = 5;

#ifdef _WIN32
    constexpr wchar_t TORCH_MARKER[] = L"torch_cpu.dll";
#else
    constexpr char TORCH_MARKER[] = "libtorch_cpu.so";
#endif

    bool IsTorchLibDirectory(const std::filesystem::path& directory)
    {
        std::error_code error;
        return std::filesystem::is_regular_file(directory / TORCH_MARKER, error);
    }

    std::optional<std::filesystem::path> FindNearbyTorchLib(
        const std::filesystem::path& launcherDirectory)
    {
        auto directory = launcherDirectory;
        for (int level = 0; level < TORCH_SEARCH_DEPTH; ++level)
        {
            auto candidate = directory / "torch-archive" / "torch" / "lib";
            if (IsTorchLibDirectory(candidate))
                return candidate;

            auto parent = directory.parent_path();
            if (parent == directory)
                break;
            directory = std::move(parent);
        }

        return std::nullopt;
    }

#ifdef _WIN32
    std::optional<std::wstring> GetEnvironmentVariableValue(const wchar_t* name)
    {
        const DWORD requiredSize = GetEnvironmentVariableW(name, nullptr, 0);
        if (requiredSize == 0)
            return std::nullopt;

        std::wstring value(requiredSize, L'\0');
        const DWORD written = GetEnvironmentVariableW(name, value.data(), requiredSize);
        if (written == 0 || written >= requiredSize)
            return std::nullopt;

        value.resize(written);
        return value;
    }

    std::optional<std::filesystem::path> FindDefaultTorchLib()
    {
        auto localAppData = GetEnvironmentVariableValue(L"LOCALAPPDATA");
        if (!localAppData)
            return std::nullopt;

        auto candidate = std::filesystem::path(*localAppData)
            / "RLBot5" / "bots" / "torch-archive" / "torch" / "lib";
        return IsTorchLibDirectory(candidate)
            ? std::optional<std::filesystem::path>(std::move(candidate))
            : std::nullopt;
    }

    std::optional<std::filesystem::path> GetLauncherPath()
    {
        std::wstring buffer(32768, L'\0');
        const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0 || length >= buffer.size())
            return std::nullopt;

        buffer.resize(length);
        return std::filesystem::path(std::move(buffer));
    }

    bool PrependTorchToPath(const std::filesystem::path& torchLib)
    {
        std::wstring updatedPath = torchLib.native();
        if (auto currentPath = GetEnvironmentVariableValue(L"PATH"); currentPath && !currentPath->empty())
            updatedPath += L";" + *currentPath;

        return SetEnvironmentVariableW(L"PATH", updatedPath.c_str()) != FALSE;
    }

    std::wstring QuoteCommandLineArgument(const std::wstring& argument)
    {
        if (!argument.empty() && argument.find_first_of(L" \t\n\v\"") == std::wstring::npos)
            return argument;

        std::wstring quoted = L"\"";
        std::size_t backslashCount = 0;
        for (const wchar_t character : argument)
        {
            if (character == L'\\')
            {
                ++backslashCount;
                continue;
            }

            if (character == L'\"')
            {
                quoted.append(backslashCount * 2 + 1, L'\\');
                quoted.push_back(L'\"');
            }
            else
            {
                quoted.append(backslashCount, L'\\');
                quoted.push_back(character);
            }
            backslashCount = 0;
        }

        quoted.append(backslashCount * 2, L'\\');
        quoted.push_back(L'\"');
        return quoted;
    }

    std::wstring BuildChildCommandLine(const std::filesystem::path& corePath)
    {
        std::wstring commandLine = QuoteCommandLineArgument(corePath.native());

        int argumentCount = 0;
        LPWSTR* arguments = CommandLineToArgvW(GetCommandLineW(), &argumentCount);
        if (!arguments)
            return commandLine;

        for (int index = 1; index < argumentCount; ++index)
        {
            commandLine.push_back(L' ');
            commandLine += QuoteCommandLineArgument(arguments[index]);
        }
        LocalFree(arguments);

        return commandLine;
    }

    void PrintWindowsError(const wchar_t* operation)
    {
        const DWORD errorCode = GetLastError();
        wchar_t* message = nullptr;
        FormatMessageW(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            nullptr,
            errorCode,
            0,
            reinterpret_cast<wchar_t*>(&message),
            0,
            nullptr);

        std::wcerr << L"GGLBot launcher: " << operation << L" failed (" << errorCode << L")";
        if (message)
        {
            std::wcerr << L": " << message;
            LocalFree(message);
        }
        else
        {
            std::wcerr << L'\n';
        }
    }

    int LaunchCore(const std::filesystem::path& launcherPath, const std::filesystem::path& corePath)
    {
        std::wstring commandLine = BuildChildCommandLine(corePath);
        std::vector<wchar_t> mutableCommandLine(commandLine.begin(), commandLine.end());
        mutableCommandLine.push_back(L'\0');

        STARTUPINFOW startupInfo{};
        startupInfo.cb = sizeof(startupInfo);
        PROCESS_INFORMATION processInfo{};

        if (!CreateProcessW(
                corePath.c_str(),
                mutableCommandLine.data(),
                nullptr,
                nullptr,
                FALSE,
                CREATE_SUSPENDED,
                nullptr,
                launcherPath.parent_path().c_str(),
                &startupInfo,
                &processInfo))
        {
            PrintWindowsError(L"starting GGLBotCore.exe");
            return EXIT_FAILURE;
        }

        // Keep the core tied to the launcher when RLBot stops the launcher process.
        HANDLE job = CreateJobObjectW(nullptr, nullptr);
        if (job)
        {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION jobInfo{};
            jobInfo.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if (!SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &jobInfo,
                    sizeof(jobInfo))
                || !AssignProcessToJobObject(job, processInfo.hProcess))
            {
                CloseHandle(job);
                job = nullptr;
            }
        }

        if (ResumeThread(processInfo.hThread) == static_cast<DWORD>(-1))
        {
            PrintWindowsError(L"resuming GGLBotCore.exe");
            TerminateProcess(processInfo.hProcess, EXIT_FAILURE);
            CloseHandle(processInfo.hThread);
            CloseHandle(processInfo.hProcess);
            if (job)
                CloseHandle(job);
            return EXIT_FAILURE;
        }

        CloseHandle(processInfo.hThread);
        WaitForSingleObject(processInfo.hProcess, INFINITE);

        DWORD exitCode = EXIT_FAILURE;
        GetExitCodeProcess(processInfo.hProcess, &exitCode);
        CloseHandle(processInfo.hProcess);
        if (job)
            CloseHandle(job);

        return static_cast<int>(exitCode);
    }
#else
    std::optional<std::filesystem::path> FindDefaultTorchLib()
    {
        std::filesystem::path dataHome;
        if (const char* xdgDataHome = std::getenv("XDG_DATA_HOME"); xdgDataHome && *xdgDataHome)
        {
            dataHome = xdgDataHome;
        }
        else if (const char* home = std::getenv("HOME"); home && *home)
        {
            dataHome = std::filesystem::path(home) / ".local" / "share";
        }
        else
        {
            return std::nullopt;
        }

        auto candidate = dataHome / "RLBot5" / "bots" / "torch-archive" / "torch" / "lib";
        return IsTorchLibDirectory(candidate)
            ? std::optional<std::filesystem::path>(std::move(candidate))
            : std::nullopt;
    }

    std::optional<std::filesystem::path> GetLauncherPath(const char* argumentZero)
    {
        std::error_code error;
        auto launcherPath = std::filesystem::read_symlink("/proc/self/exe", error);
        if (!error)
            return launcherPath;

        if (!argumentZero)
            return std::nullopt;

        launcherPath = std::filesystem::absolute(argumentZero, error);
        return error ? std::nullopt : std::optional<std::filesystem::path>(std::move(launcherPath));
    }

    bool PrependTorchToPath(const std::filesystem::path& torchLib)
    {
        std::string updatedPath = torchLib.string();
        if (const char* currentPath = std::getenv("LD_LIBRARY_PATH"); currentPath && *currentPath)
            updatedPath += ":" + std::string(currentPath);

        return setenv("LD_LIBRARY_PATH", updatedPath.c_str(), 1) == 0;
    }

    int LaunchCore(
        const std::filesystem::path& corePath,
        int argumentCount,
        char** arguments)
    {
        std::string coreArgument = corePath.string();
        std::vector<char*> childArguments;
        childArguments.reserve(static_cast<std::size_t>(argumentCount) + 1);
        childArguments.push_back(coreArgument.data());
        for (int index = 1; index < argumentCount; ++index)
            childArguments.push_back(arguments[index]);
        childArguments.push_back(nullptr);

        execv(corePath.c_str(), childArguments.data());
        std::cerr << "GGLBot launcher: starting GGLBotCore failed: "
                  << std::strerror(errno) << '\n';
        return EXIT_FAILURE;
    }
#endif
}

int main(int argc, char** argv)
{
#ifdef _WIN32
    auto launcherPath = GetLauncherPath();
#else
    auto launcherPath = GetLauncherPath(argc > 0 ? argv[0] : nullptr);
#endif
    if (!launcherPath)
    {
        std::cerr << "GGLBot launcher: could not determine the launcher path.\n";
        return EXIT_FAILURE;
    }

    const auto launcherDirectory = launcherPath->parent_path();
#ifdef _WIN32
    const auto corePath = launcherDirectory / "000-runtime" / "GGLBotCore.exe";
#else
    const auto corePath = launcherDirectory / "000-runtime" / "GGLBotCore";
#endif

    std::error_code error;
    if (!std::filesystem::is_regular_file(corePath, error))
    {
        std::cerr << "GGLBot launcher: core executable not found at " << corePath << '\n';
        return EXIT_FAILURE;
    }

    auto torchLib = FindNearbyTorchLib(launcherDirectory);
    if (!torchLib)
        torchLib = FindDefaultTorchLib();
    if (!torchLib)
    {
        std::cerr << "GGLBot launcher: could not find the RLBot torch-archive runtime.\n";
        return EXIT_FAILURE;
    }
    if (!PrependTorchToPath(*torchLib))
    {
        std::cerr << "GGLBot launcher: could not configure the Torch library path.\n";
        return EXIT_FAILURE;
    }

#ifdef _WIN32
    return LaunchCore(*launcherPath, corePath);
#else
    return LaunchCore(corePath, argc, argv);
#endif
}
