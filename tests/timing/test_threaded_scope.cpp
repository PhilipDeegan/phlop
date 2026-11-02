// tests/timing/test_threaded_scope.cpp
// PHLOP_SCOPE on several threads, each counting itself. Run with counters:
//   eval "$(python3 -m phlop.app.pfm.stat --env perf_light=light --env perf_all=ipc)"
//   ./bin/threaded_scope
//   python3 tests/timing/test_scope_timer.py test_threaded_scope -f bin/threaded_scope.txt

#include "phlop/timing/threaded_scope_timer.hpp"

#include <ctime>
#include <thread>
#include <vector>
#include <iostream>

using namespace std; // bare `time` below must still mean phlop::scope_key::time

double work(std::size_t const n, int const reps)
{
    PHLOP_SCOPE("work", time, perf_light);
    std::vector<double> a(n, 1.5), c(n, 0.0);
    for (int r = 0; r < reps; ++r)
    {
        for (std::size_t i = 0; i < n; ++i)
            c[i] = a[i] * 1.000001 + c[i];
        __asm__ volatile("" ::: "memory");
    }
    return c[n / 2];
}

int main()
{
    phlop::scope_timer().file_name("bin/threaded_scope.txt").init();
    std::vector<double> results(4);
    {
        PHLOP_SCOPE("main", time, perf_all); // counts the main thread only
        std::vector<std::thread> threads;
        for (std::size_t t = 0; t < results.size(); ++t) // thread t does (t + 1) units
            threads.emplace_back([&, t] { results[t] = work(4096, 1000 * static_cast<int>(t + 1)); });
        for (auto& t : threads)
            t.join();
    }
    phlop::scope_timer().shutdown();
    std::cout << "results " << results[0] << " " << results[3] << " at "
              << (std::time(nullptr) > 0) << '\n';
}
