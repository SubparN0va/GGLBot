#include "RLBotClient.h"

#include <rlbot/BotManager.h>

#include <filesystem>
#include <fstream>

namespace
{
    // Helper functions used to read bot.toml
    inline void ltrim(std::string& s) {
        size_t i = 0;
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) i++;
        s.erase(0, i);
    }
    inline void rtrim(std::string& s) {
        while (!s.empty()) {
            char c = s.back();
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') s.pop_back();
            else break;
        }
    }
    inline void trim(std::string& s) { ltrim(s); rtrim(s); }

    inline bool starts_with(const std::string& s, const char* prefix) {
        const size_t n = std::char_traits<char>::length(prefix);
        return s.size() >= n && s.compare(0, n, prefix) == 0;
    }

    // Reads [settings] agent_id = "..." from bot.toml
    std::optional<std::string> ReadAgentIdFromBotToml(const std::filesystem::path& botTomlPath)
    {
        std::ifstream f(botTomlPath);
        if (!f.is_open())
            return std::nullopt;

        bool inSettings = false;
        std::string line;

        while (std::getline(f, line)) {
            // Strip TOML comments (# ...)
            if (auto hash = line.find('#'); hash != std::string::npos)
                line.erase(hash);

            trim(line);
            if (line.empty()) continue;

            // Section headers
            if (line.front() == '[' && line.back() == ']') {
                inSettings = (line == "[settings]");
                continue;
            }

            if (!inSettings) continue;

            // Look for agent_id = "..."
            // Check key at start after trimming.
            if (starts_with(line, "agent_id")) {
                auto eq = line.find('=');
                if (eq == std::string::npos) continue;

                std::string rhs = line.substr(eq + 1);
                trim(rhs);

                if (rhs.empty()) continue;

                char quote = rhs.front();
                if (quote != '"' && quote != '\'') continue;

                auto endq = rhs.find(quote, 1);
                if (endq == std::string::npos) continue;

                return rhs.substr(1, endq - 1);
            }
        }

        return std::nullopt;
    }

    std::shared_ptr<const SharedBotContext>& SpawnContext() noexcept
    {
        static std::shared_ptr<const SharedBotContext> ctx;
        return ctx;
    }

    void SetSpawnContext(std::shared_ptr<const SharedBotContext> ctx) noexcept
    {
        SpawnContext() = std::move(ctx);
    }

    std::unique_ptr<rlbot::Bot> SpawnBot(std::unordered_set<unsigned> indices,
        unsigned team,
        std::string name) noexcept
    {
        auto ctx = SpawnContext();
        if (!ctx)
            return {};

        auto* p = new (std::nothrow) RLBotBot(std::move(indices), team, std::move(name), std::move(ctx));
        return std::unique_ptr<rlbot::Bot>(p);
    }

    class RLBotBotManager final : public rlbot::BotManagerBase
    {
    public:
        explicit RLBotBotManager(bool batchHivemind = false) noexcept
            : rlbot::BotManagerBase(batchHivemind, SpawnBot)
        {
        }
    };
} // anonymous namespace


int main(int argc, char** argv)
{
    auto ctx = std::make_shared<SharedBotContext>();

    // ------------------------------------------------------------------------
    // Set the following to match the configuration your model was trained with
    // ------------------------------------------------------------------------
    ctx->obs = std::make_shared<RLGC::AdvancedObs>();
    ctx->act = std::make_shared<RLGC::DefaultAction>();

    ctx->params.tickSkip = 8;
    ctx->params.actionDelay = ctx->params.tickSkip - 1;

    int obsSize = 109; // You can find this from the console when running training

    // Shared head config
    GGL::InferPartialModelConfig sharedHeadCfg;
    sharedHeadCfg.layerSizes = { 256, 256 };
    sharedHeadCfg.addLayerNorm = true;
    sharedHeadCfg.activationType = GGL::ModelActivationType::RELU;
    sharedHeadCfg.addOutputLayer = false; // <- leave this false

    // Policy config
    GGL::InferPartialModelConfig policyCfg;
    policyCfg.layerSizes = { 256, 256, 256 };
    policyCfg.addLayerNorm = true;
    policyCfg.activationType = GGL::ModelActivationType::RELU;
    policyCfg.addOutputLayer = true;

    // ------------------------------------------
    // Everything below can usually be left as is
    // ------------------------------------------
    std::filesystem::path exeDir;
    if (!argv[0]) {
        exeDir = std::filesystem::current_path();
    }
    else {
        std::filesystem::path p(argv[0]);
        exeDir = p.parent_path();
    }

    bool useGPU = false;

    ctx->inferUnit = std::make_shared<GGL::InferUnit>(
        ctx->obs.get(),
        obsSize,
        ctx->act.get(),
        sharedHeadCfg,
        policyCfg,
        exeDir, // Put model files next to exe
        useGPU
    );

    SetSpawnContext(ctx);

    auto const serverHost = []() -> char const* {
        auto const env = std::getenv("RLBOT_SERVER_IP");
        return env ? env : "127.0.0.1";
        }();

    auto const serverPort = []() -> char const* {
        auto const env = std::getenv("RLBOT_SERVER_PORT");
        return env ? env : "23234";
        }();

    // Read agent_id from bot.toml next to the exe
    const std::filesystem::path botTomlPath = exeDir / "bot.toml";
    std::string agentIdStr = "GigaLearn/GGLBot"; // fallback default
    if (auto maybeId = ReadAgentIdFromBotToml(botTomlPath)) {
        agentIdStr = *maybeId;
    }

    RLBotBotManager manager(false);

    if (!manager.connect(serverHost, serverPort, agentIdStr.c_str(), false)) {
        return EXIT_FAILURE;
    }

    return 0;
}
