// Flat C API over kag::Sim, for driving the simulator from Python via ctypes.
// No Python C API dependency -- plain cdecl exports, loaded with ctypes.CDLL.
// Deliberately exposes raw struct pointers (State*, not a copy) rather than a
// serialize-to-buffer API: fewer places to get an offset wrong, at the cost
// of requiring the Python-side ctypes.Structure mirrors in cpp_bridge.py to
// match field-for-field. sim_sizeof_* below exist so that mirror can assert
// itself correct at import time instead of silently reading garbage.
//
// Source: pyrandom.hpp / sim.hpp are an unmodified import of the C++ port
// from https://www.kaggle.com/code/nikital7/4000x-environment-speedup-kaggriculture
// (not this project's own code) -- see cpp_sim/README.md for provenance and
// this project's own validation of it.
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

// Safety-check sizes: cpp_bridge.py asserts these against ctypes.sizeof() of
// its own struct mirrors at import time, so a struct-layout mismatch (e.g.
// this file built against a different sim.hpp than the mirror was written
// against) fails loudly instead of silently misreading state.
__declspec(dllexport) int sim_sizeof_state() { return (int)sizeof(State); }
__declspec(dllexport) int sim_sizeof_farm() { return (int)sizeof(Farm); }
__declspec(dllexport) int sim_sizeof_tile() { return (int)sizeof(Tile); }
__declspec(dllexport) int sim_sizeof_market() { return (int)sizeof(Market); }
__declspec(dllexport) int sim_sizeof_action() { return (int)sizeof(Action); }
__declspec(dllexport) int sim_sizeof_config() { return (int)sizeof(Config); }

}  // extern "C"
