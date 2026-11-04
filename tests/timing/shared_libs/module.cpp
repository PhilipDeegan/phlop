// tests/timing/shared_libs/module.cpp
// like a python extension module built with -fvisibility=hidden, linked to the
// core library, scopes instantiated here: header inline statics would be its
// own hidden copies, so phlop's singletons must come from the core library

#include "phlop/timing/threaded_scope_timer.hpp"

#define EXPORT extern "C" __attribute__((visibility("default")))

EXPORT void const* module_config()
{
    return &phlop::perf::Config::INSTANCE();
}
EXPORT void const* module_man_local()
{
    return &phlop::threaded::ScopeTimerMan::local();
}
EXPORT bool module_has_perf()
{
    return phlop::perf::Config::INSTANCE().has(1);
}
EXPORT void module_work()
{
    PHLOP_SCOPE("module_scope", time, perf_light, mem);
    double volatile x = 0;
    for (int i = 0; i < 100000; ++i)
        x = x + i;
}
