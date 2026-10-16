
#include "sim.hpp"

using namespace kag;

extern "C" {

__declspec(dllexport) Sim* sim_new(const Config* cfg) {
    return new Sim(cfg ? *cfg : Config{});
}

__declspec(dllexport) void sim_free(Sim* s) {
    delete s;
}

__declspec(dllexport) void sim_step(Sim* s, const Action* a0, const Action* a1) {
    s->step(*a0, *a1);
}

__declspec(dllexport) double sim_reward(Sim* s, int player) {
    return s->reward(player);
}

__declspec(dllexport) int sim_done(Sim* s) {
    return s->st.done ? 1 : 0;
}

__declspec(dllexport) State* sim_state(Sim* s) {
    return &s->st;
}

__declspec(dllexport) int sim_sizeof_state() { return (int)sizeof(State); }
__declspec(dllexport) int sim_sizeof_farm() { return (int)sizeof(Farm); }
__declspec(dllexport) int sim_sizeof_tile() { return (int)sizeof(Tile); }
__declspec(dllexport) int sim_sizeof_market() { return (int)sizeof(Market); }
__declspec(dllexport) int sim_sizeof_action() { return (int)sizeof(Action); }
__declspec(dllexport) int sim_sizeof_config() { return (int)sizeof(Config); }

}
