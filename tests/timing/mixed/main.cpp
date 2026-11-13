// tests/timing/mixed/main.cpp: the serial phlop:: and multithread capable
//  phlop::para:: timers in one program, from separate translation units, one
//  source linked (scope_timer.cpp). Each has its own scope stack and file. See sh/test.sh

#include "phlop/timing/scope_timer.hpp"
#include "phlop/timing/para_timer.hpp" // its API, PHLOP_SCOPE here stays serial

void para_work(); // para.cpp

int main()
{
    phlop::scope_timer().file_name("bin/mixed_serial.txt").init();
    phlop::para::scope_timer().file_name("bin/mixed_para.txt").init();
    {
        PHLOP_SCOPE("serial_outer", time);
        para_work(); // para scopes do not nest under serial ones: separate timers
    }
    phlop::para::scope_timer().shutdown();
    phlop::scope_timer().shutdown();
}
