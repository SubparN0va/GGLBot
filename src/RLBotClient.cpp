#include "RLBotClient.h"

#include <algorithm>

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
        pd.isJumping = (playerInfo->air_state() == rlbot::flat::AirState::Jumping);
        pd.isFlipping = (playerInfo->air_state() == rlbot::flat::AirState::Dodging);
        pd.hasJumped = playerInfo->has_jumped();
        pd.hasDoubleJumped = playerInfo->has_double_jumped();
        pd.hasFlipped = playerInfo->has_dodged();
        pd.isDemoed = playerInfo->demolished_timeout() >= 0.f;

        // Total airtime is separate from the usable jump/flip window.
        if (pd.isOnGround) {
            timing.airTime = 0.f;
        }
        else {
            timing.airTime += dtSec;
        }

        pd.airTime = timing.airTime;
        // The packet window remains accurate after missed callbacks or a held jump.
        // A negative timeout during jump hold or after a flip reset is not an expired flip.
        if (playerInfo->dodge_timeout() >= 0.f)
            pd.airTimeSinceJump = RLConst::DOUBLEJUMP_MAX_DELAY - playerInfo->dodge_timeout();
        else if (pd.isJumping || !pd.hasJumped || pd.isOnGround)
            pd.airTimeSinceJump = 0.f;
        else
            pd.airTimeSinceJump = RLConst::DOUBLEJUMP_MAX_DELAY;

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

    const float curTime = packet->match_info()->seconds_elapsed();
    const uint32_t frame = packet->match_info()->frame_num();
    const uint32_t elapsedFrames = frame - prevFrame;
    // Unsigned subtraction handles frame-counter wrap. A backwards clock starts a new cycle.
    if (ticks >= 0 && (curTime < prevTime || elapsedFrames > 0x80000000u)) {
        ticks = -1;
        m_botState.clear();
        m_playerTiming.clear();
    }
    const bool firstPacket = ticks < 0;
    const float deltaTime = firstPacket ? 0.f : std::max(0.f, curTime - prevTime);
    prevTime = curTime;
    prevFrame = frame;

    const int tickSkip = std::max(1, ctx_->params.tickSkip);
    const int actionDelay = std::clamp(ctx_->params.actionDelay, 0, tickSkip);
    // Use physics frames so duplicate packets and float clock rounding cannot shift decisions.
    ticks = firstPacket ? 0 : ticks + static_cast<int>(std::min(elapsedFrames, uint32_t(tickSkip - ticks)));
    const bool inferAction = firstPacket || ticks >= tickSkip;
    const bool queuedActionDue = !firstPacket && ticks >= actionDelay;
    if (inferAction)
        ticks = 0;

    GameState gs = ToGameState(packet, deltaTime, m_playerTiming);

    // Every tensor row uses the same packet, with all controlled cars prepared first.
    std::vector<unsigned> activeIndices;
    for (unsigned index : indices)
        if (index < gs.players.size())
            activeIndices.push_back(index);
    std::sort(activeIndices.begin(), activeIndices.end());

    for (unsigned index : activeIndices) {
        auto& st = m_botState[index];
        // Apply the previous queued action before inference can replace it after a skipped packet.
        if (queuedActionDue && st.actionPending) {
            st.controls = st.action;
            st.actionPending = false;
        }
        gs.players[index].prevAction = st.obsPrevAction;
    }

    if (inferAction && !activeIndices.empty()) {
        const auto actions = ctx_->inferUnit->BatchInferActions(activeIndices, gs, true);
        for (size_t row = 0; row < activeIndices.size(); ++row) {
            auto& st = m_botState[activeIndices[row]];
            st.action = actions[row];
            st.obsPrevAction = st.action;
            st.actionPending = actionDelay != 0;
            if (actionDelay == 0)
                st.controls = st.action;
        }
    }

    for (unsigned index : activeIndices) {
        const auto& st = m_botState[index];
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
}
