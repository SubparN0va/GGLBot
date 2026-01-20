#pragma once

#include <rlbot/Bot.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <GigaLearnCPP/InferUnit.h>

namespace GGL { class InferUnit; }

struct RLBotParams {
    int tickSkip;
    int actionDelay;
};

struct SharedBotContext {
    std::shared_ptr<RLGC::ObsBuilder> obs;
    std::shared_ptr<RLGC::ActionParser> act;
    std::shared_ptr<GGL::InferUnit> inferUnit;
    RLBotParams params;
};

struct PlayerTimingState {
    float airTime = 0.f;
    float airTimeSinceJump = 0.f;
    bool  lastOnGround = true;
};

class RLBotBot : public rlbot::Bot {
public:
    struct PerBotState {
        bool initialized = false;

        // Queued action and current action
        RLGC::Action
            action = {},
            controls = {};
    };
    

    // Persistent info
    bool updateAction = true;
    int ticks = -1;
    float prevTime = 0;

    RLBotBot() noexcept = delete;
    ~RLBotBot() noexcept override;

    RLBotBot(std::unordered_set<unsigned> indices_,
        unsigned team_,
        std::string name_,
        std::shared_ptr<const SharedBotContext> ctx_) noexcept;

    void update(rlbot::flat::GamePacket const* packet_,
        rlbot::flat::BallPrediction const* ballPrediction_) noexcept override;

private:
    std::shared_ptr<const SharedBotContext> ctx_;
    std::unordered_map<unsigned, PerBotState> m_botState;
    std::vector<PlayerTimingState> m_playerTiming;
};
