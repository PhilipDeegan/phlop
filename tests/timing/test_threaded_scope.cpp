// tests/timing/test_threaded_scope.cpp
// PHLOP_SCOPE on several threads, each counting itself. Run with counters:
//   eval "$(python3 -m phlop.app.pfm.stat --env perf_light=light --env perf_all=ipc)"
//   ./bin/threaded_scope
//   python3 tests/timing/test_scope_timer.py test_threaded_scope -f bin/threaded_scope.txt

#include "phlop/timing/threaded_scope_timer.hpp"

#include <ctime>
#include <atomic>
#include <cstdlib>
#include <thread>
#include <vector>
#include <iostream>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace std; // bare `time` below must still mean phlop::scope_key::time

// reps / 1000 writes of 64KiB to /dev/null: per thread io, linux
double work(std::size_t const n, int const reps)
{
    PHLOP_SCOPE("work", time, perf_light, io);
#if defined(__linux__)
    std::vector<char> const buf(64 * 1024, 'x');
    int const fd = ::open("/dev/null", O_WRONLY | O_CLOEXEC);
    for (int w = 0; fd >= 0 and w < reps / 1000; ++w)
        if (::write(fd, buf.data(), buf.size()) != static_cast<ssize_t>(buf.size()))
            std::abort();
    if (fd >= 0)
        ::close(fd);
#endif
    std::vector<double> a(n, 1.5), c(n, 0.0);
    for (int r = 0; r < reps; ++r)
    {
        for (std::size_t i = 0; i < n; ++i)
            c[i] = a[i] * 1.000001 + c[i];
        std::atomic_signal_fence(std::memory_order_seq_cst); // keep each pass
    }
    return c[n / 2];
}

int main()
{
    phlop::scope_timer().file_name("bin/threaded_scope.txt").io().init();
    std::vector<double> results(4);
    {
        PHLOP_SCOPE("main", time, perf_all, io); // counts the main thread only
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
