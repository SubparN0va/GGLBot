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

void InitBotContext(std::shared_ptr<SharedBotContext> ctx);
std::shared_ptr<const SharedBotContext> GetBotContext();

class RLBotBot : public rlbot::Bot {
public:
	// Used so that the for loop in update() can actually handle multiple indices correctly
    struct PerBotState {
        bool initialized = false;

        // Position within current macro step [0, tickSkip]
        int tickInStep = 0;

        // Old action that continues applying until delay expires
        RLGC::Action prevApplied{};

        // Action computed for the current step
        RLGC::Action planned{};
        bool hasPlanned = false;

        // What we output this update
        RLGC::Action currentOut{};
    };

    std::unordered_map<unsigned, PerBotState> m_botState;

    // Packet-to-packet tick tracking
    bool m_hasPrevFrame = false;
    uint32_t m_prevFrame = 0;

    RLBotBot() noexcept = delete;
    ~RLBotBot() noexcept override;
    RLBotBot(std::unordered_set<unsigned> indices_, unsigned team_, std::string name_) noexcept;

    RLBotBot(RLBotBot const&) noexcept = delete;
    RLBotBot(RLBotBot&&) noexcept = delete;
    RLBotBot& operator=(RLBotBot const&) noexcept = delete;
    RLBotBot& operator=(RLBotBot&&) noexcept = delete;

    void update(rlbot::flat::GamePacket const* packet_,
        rlbot::flat::BallPrediction const* ballPrediction_) noexcept override;

private:
    std::shared_ptr<const SharedBotContext> ctx_;
};