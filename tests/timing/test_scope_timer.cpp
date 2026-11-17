#include "phlop/timing/scope_timer.hpp"

#include "scope_timer_fns.hpp"

#include <iostream>

int main()
{
    std::cout << __FILE__ << std::endl;
    // ONLY SUPPORTS SERIAL OPERATIONS!
    phlop::scope_timer().file_name("bin/scope_timer.txt").memory().io().init();

    run_scope_timer_fns("bin/scope_timer_io.bin");

    phlop::scope_timer().shutdown();
}
