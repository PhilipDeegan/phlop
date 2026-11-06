
#include "phlop/timing/scope_timer.hpp"

#include <thread>
#include <vector>
#include <cstdlib>
#include <iostream>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace std::chrono_literals;

void fny()
{
    PHLOP_SCOPE_TIMER("fny");

    std::this_thread::sleep_for(10ms);
}

void fn0()
{
    PHLOP_SCOPE_TIMER("fn0");

    std::this_thread::sleep_for(10ms);

    fny();

    PHLOP_SCOPE_TIMER("fn0b");
}

void fn1()
{
    PHLOP_SCOPE_TIMER("fn1");

    std::this_thread::sleep_for(100ms);
    PHLOP_SCOPE_TIMER("fn1b");
    fn0();
}

void fn2()
{
    PHLOP_SCOPE_TIMER("fn2");

    std::this_thread::sleep_for(200ms);
    fn0();
}

// make bytes resident, then free them
void touch(std::size_t const bytes)
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

void fn_mem()
{
    PHLOP_SCOPE("fn_mem", time, mem);

    touch(64 * 1024 * 1024);
}

void fn_peak_big()
{
    PHLOP_SCOPE("fn_peak_big", time, mem);

    touch(256 * 1024 * 1024);
}

void fn_peak_small()
{
    PHLOP_SCOPE("fn_peak_small", time, mem);

    touch(32 * 1024 * 1024); // below the earlier high, only seen with a peak reset
}

void fn_peak()
{
    PHLOP_SCOPE("fn_peak", time, mem); // its peak must include fn_peak_big's

    fn_peak_big();
    fn_peak_small();
}

// exact io: 4 writes of 1MiB, one read of 1MiB back. perf_light too, so the
//  scope's own counter reads (with PHLOP_PERF_LIGHT set) must not show
void fn_io()
{
#if defined(__linux__)
    std::size_t constexpr MiB = 1024 * 1024;
    std::vector<char> buf(MiB, 'x');
    int const fd = ::open("bin/scope_timer_io.bin", O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
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
    ::unlink("bin/scope_timer_io.bin");
#endif
}

void fn3()
{
    PHLOP_SCOPE_TIMER("fn3"); // time only scope, with a mem scope child

    fn_mem();
}

int main()
{
    std::cout << __FILE__ << std::endl;
    // ONLY SUPPORTS SERIAL OPERATIONS!
    phlop::scope_timer().file_name("bin/scope_timer.txt").memory().io().init();

    for (std::size_t i = 0; i < 2; ++i)
        fn1();
    for (std::size_t i = 0; i < 2; ++i)
        fn2();
    for (std::size_t i = 0; i < 2; ++i)
        fny();

    fn3();
    fn_peak();
    fn_io();

    phlop::scope_timer().shutdown();
}
