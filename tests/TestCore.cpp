#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    const bool cuda = std::string(TEST_DEVICE) == "CUDA";
    if (const char* trace = std::getenv("GGLBOT_TEST_TRACE"))
        std::ofstream(trace, std::ios::app) << TEST_DEVICE << '\n';

    // Verify arguments survive the launcher's Windows command-line quoting.
    if (const char* expected = std::getenv("GGLBOT_TEST_ARGUMENT")) {
        if (argc < 2 || std::string(argv[argc - 1]) != expected)
            return 98;
    }
    const char* expectedRuntime = std::getenv(cuda ? "GGLBOT_TEST_CUDA_PATH" : "GGLBOT_TEST_CPU_PATH");
#ifdef _WIN32
    const char* searchPath = std::getenv("PATH");
    constexpr char separator = ';';
#else
    const char* searchPath = std::getenv("LD_LIBRARY_PATH");
    constexpr char separator = ':';
#endif
    if (expectedRuntime && (!searchPath || std::string(searchPath).substr(0, std::string(searchPath).find(separator)) != expectedRuntime))
        return 97;
    if (!cuda) {
        if (const char* cudaPath = std::getenv("GGLBOT_TEST_CUDA_PATH");
            cudaPath && searchPath && std::string(searchPath).find(cudaPath) != std::string::npos)
            return 96;
    }
    if (cuda) {
        if (const char* status = std::getenv("GGLBOT_TEST_CUDA_EXIT")) {
            const int result = static_cast<int>(std::strtoul(status, nullptr, 0));
            if (result != 0) {
                std::cerr << "GGLBot: GPU initialization failed: simulated unavailable CUDA.\n";
                return result;
            }
        }
    }
    std::cout << "GGLBot: using " << (cuda ? "GPU" : "CPU") << '\n';
    return 0;
}
