// tests/timing/test_scope.cpp
// PHLOP_SCOPE with time and perf levels. Run with counters:
//   eval "$(python3 -m phlop.app.pfm.stat --env perf_light=light --env perf_all=flops)"
//   ./bin/scope && python3 tests/timing/test_scope_timer.py test_scope -f bin/scope.txt

#include "phlop/timing/scope_timer.hpp"

#include <ctime>
#include <atomic>
#include <vector>
#include <iostream>

using namespace std; // bare `time` below must still mean phlop::scope_key::time

static_assert(phlop::scope_flags() == 1); // PHLOP_SCOPE_DEFAULT is time
static_assert(sizeof(phlop::Scope<1>) == sizeof(phlop::ScopeTimer<>)); // time only: no extra

double fma_loop(std::vector<double>& c, std::vector<double> const& a, int const reps)
{
    PHLOP_SCOPE("fma_loop", time, perf_all);
    for (int r = 0; r < reps; ++r)
    {
        PHLOP_SCOPE("fma_pass", time, perf_light);
        for (std::size_t i = 0; i < c.size(); ++i)
            c[i] = a[i] * 1.000001 + c[i];
        std::atomic_signal_fence(std::memory_order_seq_cst); // keep each pass
    }
    return c[c.size() / 2];
}

int main()
{
    phlop::scope_timer().file_name("bin/scope.txt").init();
    double result = 0;
    {
        PHLOP_SCOPE("main"); // default flags
        std::vector<double> a(4096, 1.5), c(4096, 0.0);
        for (int i = 0; i < 3; ++i)
            result += fma_loop(c, a, 2000);
        PHLOP_SCOPE("tail", time);
    }
    phlop::scope_timer().shutdown();
    std::cout << "result " << result << " at " << (std::time(nullptr) > 0) << '\n';
}
