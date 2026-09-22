#include "RLBotClient.h"
#include <GigaLearnCPP/Models.h>

#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace flat = rlbot::flat;
using namespace RLGC;

namespace {
    void Check(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }

    void Near(float actual, float expected, const char* message) {
        Check(std::abs(actual - expected) < 1e-5f, message);
    }

    struct Observation {
        Player player;
        std::vector<float> previousActions;
        const GameState* source;
    };

    struct TestObs final : ObsBuilder {
        std::vector<Observation> seen;
        FList BuildObs(const Player& player, const GameState& state) override {
            Observation observation{player, {}, &state};
            for (const auto& other : state.players)
                observation.previousActions.push_back(other.prevAction.throttle);
            seen.push_back(std::move(observation));
            return {float(player.carId), player.prevAction.throttle, float(player.HasFlipOrJump())};
        }
    };

    struct TestActions final : ActionParser {
        int GetActionAmount() override { return 3; }
        std::vector<uint8_t> GetActionMask(const Player& player, const GameState&) override {
            std::vector<uint8_t> mask(3, 0);
            mask[(player.carId - 1 + static_cast<unsigned>(player.boost)) % 3] = 1;
            return mask;
        }
        Action ParseAction(int index, const Player&, const GameState&) override {
            Action action{};
            action.throttle = (index + 1) / 3.f;
            return action;
        }
    };

    // Observe actual tensor shapes without replacing the model or its inference path.
    struct BatchRecorder : torch::nn::Module {
        std::vector<int64_t>* batches;
        explicit BatchRecorder(std::vector<int64_t>* sizes) : batches(sizes) {}
        torch::Tensor forward(torch::Tensor input) {
            batches->push_back(input.size(0));
            return input;
        }
    };

    struct Fixture {
        std::shared_ptr<TestObs> obs = std::make_shared<TestObs>();
        std::shared_ptr<TestActions> parser = std::make_shared<TestActions>();
        std::vector<int64_t> batches;
        std::shared_ptr<SharedBotContext> ctx = std::make_shared<SharedBotContext>();

        explicit Fixture(const std::filesystem::path& models) {
            GGL::InferPartialModelConfig policy;
            policy.layerSizes = {8};
            policy.addLayerNorm = false;
            GGL::ModelConfig config(policy);
            config.numInputs = 3;
            config.numOutputs = 3;
            GGL::Model model("policy", config, torch::kCPU);
            std::filesystem::create_directories(models);
            torch::save(model.seq, (models / "POLICY.lt").string());
            ctx->obs = obs;
            ctx->act = parser;
            ctx->params = {8, 7};
            ctx->inferUnit = std::make_shared<GGL::InferUnit>(
                obs.get(), 3, parser.get(), GGL::InferPartialModelConfig{}, policy,
                models, GGLBOT_USE_CUDA != 0);
            (*ctx->inferUnit->models)["policy"]->seq->push_back(std::make_shared<BatchRecorder>(&batches));
        }
    };

    struct Packet {
        flat::GamePacketT data;
        explicit Packet(int count = 6) {
            data.match_info = std::make_unique<flat::MatchInfoT>();
            data.boost_pads.assign(CommonValues::BOOST_LOCATIONS_AMOUNT, flat::BoostPadState(true, 0.f));
            auto ball = std::make_unique<flat::BallInfoT>();
            ball->physics = std::make_unique<flat::Physics>();
            data.balls.push_back(std::move(ball));
            for (int i = 0; i < count; ++i) {
                auto player = std::make_unique<flat::PlayerInfoT>();
                player->physics = std::make_unique<flat::Physics>();
                player->player_id = i + 1;
                player->team = i < 3 ? 0 : 1;
                player->boost = 0;
                player->air_state = flat::AirState::OnGround;
                player->demolished_timeout = -1;
                player->dodge_timeout = -1;
                data.players.push_back(std::move(player));
            }
        }

        void Variant(unsigned value) {
            for (auto& player : data.players)
                player->boost = float(value);
        }

        void Send(RLBotBot& bot, uint32_t frame, float seconds = -1.f) {
            data.match_info->frame_num = frame;
            data.match_info->seconds_elapsed = seconds < 0 ? frame / 120.f : seconds;
            flatbuffers::FlatBufferBuilder buffer;
            buffer.Finish(flat::GamePacket::Pack(buffer, &data));
            bot.update(flatbuffers::GetRoot<flat::GamePacket>(buffer.GetBufferPointer()), nullptr);
        }
    };

    void Controls(RLBotBot& bot, unsigned variant, unsigned count = 3) {
        for (unsigned index = 0; index < count; ++index)
            Near(bot.getOutput(index).throttle(), ((index + variant) % 3 + 1) / 3.f, "wrong car/action mapping");
    }

    void Neutral(RLBotBot& bot) {
        for (unsigned index : bot.indices)
            Near(bot.getOutput(index).throttle(), 0.f, "action applied before its delay");
    }
}

int main(int argc, char** argv) {
    if (argc != 2)
        return 1;
    torch::set_num_threads(1);
    const std::filesystem::path models = argv[1];
    int passed = 0;
    auto run = [&](const char* name, const std::function<void(Fixture&)>& test) {
        Fixture fixture(models);
        test(fixture);
        ++passed;
        std::cout << "PASS " << name << '\n';
    };

    try {
        run("batch rows, masks, singleton parity and shared state", [](Fixture& f) {
            GameState state;
            state.players.resize(6);
            for (unsigned i = 0; i < 6; ++i) {
                state.players[i].carId = i + 1;
                state.players[i].boost = 0;
                state.players[i].prevAction.throttle = (i + 1) / 10.f;
            }
            const std::vector<unsigned> indices{5, 1, 3};
            const auto actions = f.ctx->inferUnit->BatchInferActions(indices, state, true);
            Check(f.batches == std::vector<int64_t>{3}, "expected one forward with three rows");
            for (size_t row = 0; row < indices.size(); ++row) {
                Check(f.obs->seen[row].source == &state, "batch copied its game state");
                Check(f.obs->seen[row].player.carId == indices[row] + 1, "row order changed");
                Near(actions[row].throttle, (indices[row] % 3 + 1) / 3.f, "mask applied to wrong row");
                const auto single = f.ctx->inferUnit->InferAction(state.players[indices[row]], state, true);
                Near(actions[row].throttle, single.throttle, "batch differs from singleton");
            }
            const auto legacy = f.ctx->inferUnit->BatchInferActions(
                std::vector<Player>{state.players[5], state.players[1]}, std::vector<GameState>{state, state}, true);
            Near(legacy[0].throttle, actions[0].throttle, "legacy batch changed");
        });

        run("empty and invalid batches", [](Fixture& f) {
            GameState state;
            Check(f.ctx->inferUnit->BatchInferActions(std::vector<unsigned>{}, state, true).empty(), "empty batch failed");
            bool rejected = false;
            try { f.ctx->inferUnit->BatchInferActions(std::vector<unsigned>{0}, state, true); }
            catch (const std::out_of_range&) { rejected = true; }
            Check(rejected && f.batches.empty(), "invalid batch ran inference");
        });

        run("decision cadence and per-car delayed controls", [](Fixture& f) {
            RLBotBot bot({2, 0, 1}, 0, "test", f.ctx);
            Packet packet;
            for (unsigned frame = 0; frame <= 6; ++frame) {
                packet.Send(bot, frame);
                Neutral(bot);
            }
            Check(f.batches == std::vector<int64_t>{3}, "extra startup inference");
            packet.Send(bot, 7);
            Controls(bot, 0);
            packet.Variant(1);
            packet.Send(bot, 8);
            Check(f.batches == std::vector<int64_t>({3, 3}), "decision missed frame 8");
            Controls(bot, 0);
            for (unsigned row = 3; row < 6; ++row)
                for (unsigned car = 0; car < 3; ++car)
                    Near(f.obs->seen[row].previousActions[car], (car + 1) / 3.f, "teammate history was not prepared");
            packet.Send(bot, 14);
            Controls(bot, 0);
            packet.Send(bot, 15);
            Controls(bot, 1);
            packet.Send(bot, 16);
            packet.Send(bot, 24);
            Check(f.batches.size() == 4, "policy did not run at 0, 8, 16, 24");
        });

        run("zero delay, duplicate packets and rounded match clock", [](Fixture& f) {
            f.ctx->params.actionDelay = 0;
            RLBotBot bot({0, 1, 2}, 0, "test", f.ctx);
            Packet packet;
            packet.Send(bot, 12000, 1000000.f);
            Controls(bot, 0);
            packet.Variant(1);
            packet.Send(bot, 12001, 1000000.f);
            packet.Send(bot, 12001, 1000000.f);
            Check(f.batches.size() == 1, "duplicate/rounded clock caused inference");
            packet.Send(bot, 12008, 1000000.f);
            Controls(bot, 1);
            Check(f.batches.size() == 2, "frame count lost to clock rounding");
        });

        run("skipped packet activates old action before new decision", [](Fixture& f) {
            RLBotBot bot({0, 1, 2}, 0, "test", f.ctx);
            Packet packet;
            packet.Send(bot, 0);
            packet.Variant(1);
            packet.Send(bot, 8);
            Controls(bot, 0);
            packet.Send(bot, 15);
            Controls(bot, 1);
            packet.Variant(2);
            packet.Send(bot, 100);
            Check(f.batches.size() == 3, "long gap replayed stale decisions");
            Controls(bot, 1);
            packet.Send(bot, 107);
            Controls(bot, 2);
        });

        run("delay equal to tick skip retains previous policy action", [](Fixture& f) {
            f.ctx->params.actionDelay = 8;
            RLBotBot bot({0, 1, 2}, 0, "test", f.ctx);
            Packet packet;
            packet.Send(bot, 0);
            Neutral(bot);
            packet.Variant(1);
            packet.Send(bot, 8);
            Controls(bot, 0);
            Near(f.obs->seen[3].player.prevAction.throttle, 1.f / 3, "observed stale held controls");
            packet.Send(bot, 16);
            Controls(bot, 1);
        });

        run("clock rewind resets queued controls and cadence", [](Fixture& f) {
            RLBotBot bot({0, 1, 2}, 0, "test", f.ctx);
            Packet packet;
            packet.Send(bot, 1000);
            packet.Send(bot, 1007);
            Controls(bot, 0);
            packet.Variant(1);
            packet.Send(bot, 0);
            Neutral(bot);
            Near(f.obs->seen[3].player.prevAction.throttle, 0.f, "old match action survived reset");
            packet.Send(bot, 7);
            Controls(bot, 1);
        });

        run("unsigned physics-frame wrap", [](Fixture& f) {
            RLBotBot bot({0, 1, 2}, 0, "test", f.ctx);
            Packet packet;
            packet.Send(bot, std::numeric_limits<uint32_t>::max() - 3, 100.f);
            packet.Send(bot, 4, 100.f + 8.f / 120);
            Controls(bot, 0);
            Check(f.batches.size() == 2, "frame wrap broke cadence");
        });

        run("batch shrinks to currently available cars", [](Fixture& f) {
            f.ctx->params.actionDelay = 0;
            RLBotBot bot({0, 1, 2}, 0, "test", f.ctx);
            Packet packet;
            packet.Send(bot, 0);
            packet.data.players.resize(2);
            packet.Send(bot, 8);
            Check(f.batches == std::vector<int64_t>({3, 2}), "wrong batch after car removal");
            Controls(bot, 0, 2);
            packet.data.players.clear();
            packet.Send(bot, 16);
            Check(f.batches.size() == 2, "ran an empty policy batch");
        });

        run("reported flip window, jump hold, reset and spent jumps", [](Fixture& f) {
            f.ctx->params = {1, 0};
            RLBotBot bot({0}, 0, "test", f.ctx);
            Packet packet(1);
            DefaultAction actions;
            struct Case { flat::AirState air; float timeout; bool jumped, doubled, dodged, available; };
            const Case cases[] = {
                {flat::AirState::Jumping, -1.f, true, false, false, true},
                {flat::AirState::InAir, .05f, true, false, false, true},
                {flat::AirState::InAir, 0.f, true, false, false, false},
                {flat::AirState::InAir, -1.f, true, false, false, false},
                {flat::AirState::InAir, -1.f, false, false, false, true},
                {flat::AirState::DoubleJumping, .9f, true, true, false, false},
                {flat::AirState::Dodging, .9f, true, false, true, false},
                {flat::AirState::OnGround, -1.f, false, false, false, true},
            };
            unsigned frame = 0;
            for (const auto& value : cases) {
                auto& raw = *packet.data.players[0];
                raw.air_state = value.air;
                raw.dodge_timeout = value.timeout;
                raw.has_jumped = value.jumped;
                raw.has_double_jumped = value.doubled;
                raw.has_dodged = value.dodged;
                packet.Send(bot, frame, 300.f + frame / 120.f);
                ++frame;
                const auto& player = f.obs->seen.back().player;
                Check(player.HasFlipOrJump() == value.available, "incorrect flip availability");
                Check(player.isJumping == (value.air == flat::AirState::Jumping), "jump phase missing");
                GameState state;
                const auto mask = actions.GetActionMask(player, state);
                bool canJump = false;
                for (int i = 0; i < actions.GetActionAmount(); ++i)
                    if (mask[i] && actions.ParseAction(i, player, state).jump > .5f)
                        canJump = true;
                Check(canJump == value.available, "jump mask differs from reported window");
                if (value.timeout > 0.f)
                    Near(player.airTimeSinceJump, RocketSim::RLConst::DOUBLEJUMP_MAX_DELAY - value.timeout, "used accumulated airtime");
            }
            Near(f.obs->seen.front().player.airTime, 0.f, "first packet counted elapsed match time as airtime");
        });
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL after " << passed << " checks: " << error.what() << '\n';
        return 1;
    }
    std::cout << passed << " runtime checks passed on " << (GGLBOT_USE_CUDA ? "GPU" : "CPU") << '\n';
    return 0;
}
