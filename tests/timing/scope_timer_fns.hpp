#ifndef _PHLOP_TESTS_TIMING_SCOPE_TIMER_FNS_HPP_
#define _PHLOP_TESTS_TIMING_SCOPE_TIMER_FNS_HPP_

// the single threaded scopes checked by test_scope_timer.py test_scope_timer,
//  shared by test_scope_timer.cpp and test_mpi_scope_timer.cpp. Include a timer
//  header first, for PHLOP_SCOPE/PHLOP_SCOPE_TIMER

#include <string>
#include <thread>
#include <vector>
#include <cstdlib>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace std::chrono_literals;

inline void fny()
{
    PHLOP_SCOPE_TIMER("fny");

    std::this_thread::sleep_for(10ms);
}

inline void fn0()
{
    PHLOP_SCOPE_TIMER("fn0");

    std::this_thread::sleep_for(10ms);

    fny();

    PHLOP_SCOPE_TIMER("fn0b");
}

inline void fn1()
{
    PHLOP_SCOPE_TIMER("fn1");

    std::this_thread::sleep_for(100ms);
    PHLOP_SCOPE_TIMER("fn1b");
    fn0();
}

inline void fn2()
{
    PHLOP_SCOPE_TIMER("fn2");

    std::this_thread::sleep_for(200ms);
    fn0();
}

// make bytes resident, then free them
inline void touch(std::size_t const bytes)
{
    std::vector<char> v(bytes, 1); // value initialized, so resident

    // read back through volatile so the optimizer can't elide the allocation
    //  or its writes (clang -O2 removes the whole vector otherwise)
    char const volatile* const p = v.data();
    std::size_t sum              = 0;
    for (std::size_t i = 0; i < v.size(); i += 4096)
        sum += static_cast<std::size_t>(p[i]);
    if (sum != v.size() / 4096)
        std::abort();
}

inline void fn_mem()
{
    PHLOP_SCOPE("fn_mem", time, mem);

    touch(64 * 1024 * 1024);
}

inline void fn_peak_big()
{
    PHLOP_SCOPE("fn_peak_big", time, mem);

    touch(256 * 1024 * 1024);
}

inline void fn_peak_small()
{
    PHLOP_SCOPE("fn_peak_small", time, mem);

    touch(32 * 1024 * 1024); // below the earlier high, only seen with a peak reset
}

inline void fn_peak()
{
    PHLOP_SCOPE("fn_peak", time, mem); // its peak must include fn_peak_big's

    fn_peak_big();
    fn_peak_small();
}

// exact io: 4 writes of 1MiB, one read of 1MiB back. perf_light too, so the
//  scope's own counter reads (with PHLOP_PERF_LIGHT set) must not show
inline void fn_io([[maybe_unused]] std::string const& path)
{
#if defined(__linux__)
    std::size_t constexpr MiB = 1024 * 1024;
    std::vector<char> buf(MiB, 'x');
    int const fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0)
        std::abort();
    {
        PHLOP_SCOPE("fn_io", time, perf_light, io);
        for (int i = 0; i < 4; ++i)
            if (::write(fd, buf.data(), MiB) != static_cast<ssize_t>(MiB))
                std::abort();
        if (::pread(fd, buf.data(), MiB, 0) != static_cast<ssize_t>(MiB))
            std::abort();
    }
    ::close(fd);
    ::unlink(path.c_str());
#endif
}

inline void fn3()
{
    PHLOP_SCOPE_TIMER("fn3"); // time only scope, with a mem scope child

    fn_mem();
}

// the scopes in the order test_scope_timer expects, fn1 first
inline void run_scope_timer_fns(std::string const& io_path)
{
    for (std::size_t i = 0; i < 2; ++i)
        fn1();
    for (std::size_t i = 0; i < 2; ++i)
        fn2();
    for (std::size_t i = 0; i < 2; ++i)
        fny();

    fn3();
    fn_peak();
    fn_io(io_path);
}

#endif /*_PHLOP_TESTS_TIMING_SCOPE_TIMER_FNS_HPP_*/
