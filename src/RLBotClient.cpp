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

    Player ToPlayer(const rlbot::flat::PlayerInfo* playerInfo, float dtSec, PlayerTimingState& timing)
    {
        Player pd = {};

        static_cast<PhysState&>(pd) = ToPhysObj(playerInfo->physics());

        pd.carId = playerInfo->player_id();
        pd.team = (Team)playerInfo->team();

        pd.boost = playerInfo->boost();

        pd.isOnGround = (playerInfo->air_state() == rlbot::flat::AirState::OnGround);
        pd.hasJumped = playerInfo->has_jumped();
        pd.hasDoubleJumped = playerInfo->has_double_jumped();
        pd.hasFlipped = playerInfo->has_dodged();
        pd.isDemoed = playerInfo->demolished_timeout() >= 0.f;

        // Approximate airtime timers (used for HasFlipOrJump behavior)
        if (pd.isOnGround) {
            timing.airTime = 0.f;
            timing.airTimeSinceJump = 0.f;
        }
        else {
            timing.airTime += dtSec;

            timing.airTimeSinceJump = pd.hasJumped ? timing.airTime : 0.f;
        }

        pd.airTime = timing.airTime;
        pd.airTimeSinceJump = timing.airTimeSinceJump;

        return pd;
    }

    GameState ToGameState(rlbot::flat::GamePacket const* packet, float dtSec, std::vector<PlayerTimingState>& playerTiming) {
        GameState gs = {};

        auto players = packet->players();
        if (players) {
            const int n = (int)players->size();
            if ((int)playerTiming.size() < n)
                playerTiming.resize(n);

            gs.players.reserve(n);
            for (int i = 0; i < n; i++) {
                gs.players.push_back(ToPlayer(players->Get(i), dtSec, playerTiming[i]));
            }
        }

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

                gs.boostPadTimers[i] = boostPadStates->Get(i)->timer();
                gs.boostPadTimersInv[CommonValues::BOOST_LOCATIONS_AMOUNT - i - 1] = gs.boostPadTimers[i];
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
    if (!packet || !packet->match_info() || !packet->balls() || packet->balls()->size() == 0) {
        return;
    }

    float curTime = packet->match_info()->seconds_elapsed();
    float deltaTime = curTime - prevTime;
    prevTime = curTime;

    int ticksElapsed = roundf(deltaTime * 120);
    ticks += ticksElapsed;

    GameState gs = ToGameState(packet, deltaTime, m_playerTiming);
    
    for (auto const& index : this->indices)
    {
        auto& st = m_botState[index];
        if (!st.initialized) {
            st.initialized = true;

            st.action = RLGC::Action{};
            st.controls = RLGC::Action{};
        }

        auto& localPlayer = gs.players[index];
        localPlayer.prevAction = st.controls;

        if (updateAction) {
            st.action = ctx_->inferUnit->InferAction(localPlayer, gs, true);
        }

        if (ticks >= (ctx_->params.actionDelay) || ticks == -1) {
            // Apply new action
            st.controls = st.action;
        }

        const auto& c = st.controls;
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

    if (updateAction) {
        updateAction = false;
    }

    if (ticks >= ctx_->params.tickSkip || ticks == -1) {
        // Trigger action update next tick
        ticks = 0;
        updateAction = true;
    }
}
