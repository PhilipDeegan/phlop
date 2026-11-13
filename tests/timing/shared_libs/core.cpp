// tests/timing/shared_libs/core.cpp
// like a project's core shared library: phlop's .cpp is linked in here and
// the timer is initialised from here, see sh/test_shared_libs.sh

#include "phlop/timing/para_scope_timer.hpp"

#include <cstdlib>

extern "C" void core_init()
{
    phlop::para::scope_timer().file_name(std::getenv("PHLOP_SHARED_LIBS_OUT")).init();
}
extern "C" void core_shutdown()
{
    phlop::para::scope_timer().shutdown();
}
extern "C" void const* core_config()
{
    return &phlop::perf::Config::INSTANCE();
}
extern "C" void const* core_man_local()
{
    return &phlop::para::ScopeTimerMan::local();
}
extern "C" void core_outer(void (*inner)())
{
    PHLOP_SCOPE("core_outer", time); // inner, from the module, must nest under this
    inner();
}
