#include "RLBotClient.h"

using namespace RLGC;

namespace
{
    Vec ToVec(const rlbot::flat::Vector3 rlbotVec) {
        return Vec(rlbotVec.x(), rlbotVec.y(), rlbotVec.z());
    }

    PhysState ToPhysObj(const rlbot::flat::Physics* phys) {
        PhysState obj = {};
        obj.pos = ToVec(phys->location());

        Angle ang = Angle(phys->rotation().yaw(), phys->rotation().pitch(), phys->rotation().roll());
        obj.rotMat = ang.ToRotMat();

        obj.vel = ToVec(phys->velocity());
        obj.angVel = ToVec(phys->angular_velocity());

        return obj;
    }

    Player ToPlayer(const rlbot::flat::PlayerInfo* playerInfo) {
        Player pd = {};

        static_cast<PhysState&>(pd) = ToPhysObj(playerInfo->physics());

        pd.carId = playerInfo->player_id();
        pd.team = (Team)playerInfo->team();

        pd.boost = playerInfo->boost();
        pd.isOnGround = playerInfo->air_state() == rlbot::flat::AirState::OnGround;
        pd.hasJumped = playerInfo->has_jumped();
        pd.hasDoubleJumped = playerInfo->has_double_jumped();
        pd.isDemoed = playerInfo->demolished_timeout() >= 0;

        return pd;
    }

    GameState ToGameState(rlbot::flat::GamePacket const* packet) {
        GameState gs = {};

        auto players = packet->players();
        for (int i = 0; i < players->size(); i++)
            gs.players.push_back(ToPlayer(players->Get(i)));

        static_cast<PhysState&>(gs.ball) = ToPhysObj(packet->balls()->Get(0)->physics());

        auto boostPadStates = packet->boost_pads();
        if (boostPadStates->size() != CommonValues::BOOST_LOCATIONS_AMOUNT) {
            if (rand() % 20 == 0) { // Don't spam-log as that will lag the bot
                RG_LOG(
                    "RLBotClient ToGameState(): Bad boost pad amount, expected "
                    << CommonValues::BOOST_LOCATIONS_AMOUNT << " but got " << boostPadStates->size()
                );
            }

            // Just set all boost pads to on
            std::fill(gs.boostPads.begin(), gs.boostPads.end(), 1);
        }
        else {
            for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
                gs.boostPads[i] = boostPadStates->Get(i)->is_active();
                gs.boostPadsInv[CommonValues::BOOST_LOCATIONS_AMOUNT - i - 1] = gs.boostPads[i];
            }
        }

        return gs;
    }
} // anonymous namespace

RLBotBot::RLBotBot(std::unordered_set<unsigned> indices_,
    unsigned const team_,
    std::string name_,
    std::shared_ptr<const SharedBotContext> ctx) noexcept
    : rlbot::Bot(std::move(indices_), team_, std::move(name_))
    , ctx_(std::move(ctx))
{
    std::set<unsigned> sorted(std::begin(indices), std::end(indices));
    for (auto const& index : sorted)
        std::printf("Team %u Index %u: %s created\n", team_, index, name_.c_str());
}

RLBotBot::~RLBotBot() {}

void RLBotBot::update(rlbot::flat::GamePacket const* packet,
    rlbot::flat::BallPrediction const* ballPrediction_) noexcept
{
    if (!packet || !packet->match_info() || !packet->balls()) {
        for (auto const& index : this->indices) setOutput(index, {});
        return;
    }

    // If there's no ball, there's nothing to chase; output zeros for all
    if (packet->balls()->size() == 0) {
        for (auto const& index : this->indices) setOutput(index, {});
        return;
    }

    // Build GameState once per packet
    GameState gs = ToGameState(packet);

    uint32_t curFrame = packet->match_info()->frame_num();

    uint32_t rawDt = 1;
    if (m_hasPrevFrame) {
        rawDt = curFrame - m_prevFrame;
        if (rawDt == 0) rawDt = 1;
    }

    uint32_t dtFrames = rawDt;

    // Treat very large jumps as a discontinuity/reset
    if (dtFrames > 240) {
        dtFrames = 1;
        m_botState.clear();
    }

    m_prevFrame = curFrame;
    m_hasPrevFrame = true;

    const int tickSkip = std::max(1, ctx_->params.tickSkip);
    int actionDelay = ctx_->params.actionDelay;
    if (actionDelay < 0) actionDelay = 0;
    if (actionDelay > tickSkip) actionDelay = tickSkip;

    // Process each controlled player index independently
    for (auto const& index : this->indices)
    {
        // If we're not in the game packet; output zeros
        if (!packet->players() || packet->players()->size() <= index) {
            setOutput(index, {});
            continue;
        }
        if ((int)gs.players.size() <= (int)index) {
            setOutput(index, {});
            continue;
        }

        // Get per-bot state
        auto& st = m_botState[index];
        if (!st.initialized) {
            st.initialized = true;
            st.tickInStep = 0;
            st.prevApplied = RLGC::Action{};
            st.planned = RLGC::Action{};
            st.hasPlanned = false;
            st.currentOut = RLGC::Action{};
        }

        // Advance by dtFrames ticks
        // If dtFrames > 1, we don't have intermediate obs, so keep the same planned action for those missing ticks.
        for (uint32_t k = 0; k < dtFrames; ++k) {
            // At the start of each macro-step, compute a new planned action once
            if (st.tickInStep == 0) {
                auto& localPlayer = gs.players[index];
                localPlayer.prevAction = st.prevApplied;
                st.planned = ctx_->inferUnit->InferAction(localPlayer, gs, true);
                st.hasPlanned = true;
            }

            if (st.tickInStep < actionDelay || !st.hasPlanned) {
                st.currentOut = st.prevApplied;
            }
            else {
                st.currentOut = st.planned;
            }

            // Advance within the macro-step
            st.tickInStep++;

            // Macro-step boundary
            if (st.tickInStep >= tickSkip) {
                st.tickInStep = 0;

                // Planned action becomes the old action for next step
                if (st.hasPlanned) {
                    st.prevApplied = st.planned;
                }
                st.hasPlanned = false;
            }
        }

        const auto& c = st.currentOut;
        setOutput(index, {
            c.throttle,
            c.steer,
            c.pitch,
            c.yaw,
            c.roll,
            c.jump > 0.5f,
            c.boost > 0.5f,
            c.handbrake > 0.5f,
            false,
            });
    }
}
