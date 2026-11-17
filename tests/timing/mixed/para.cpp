// tests/timing/mixed/para.cpp: PHLOP_SCOPE into phlop::para::, see main.cpp

#include "phlop/timing/para_scope_timer.hpp"

#include <thread>

void para_work()
{
    PHLOP_SCOPE("para_outer", time);
    std::thread{[] { PHLOP_SCOPE("para_thread", time); }}.join();
}
