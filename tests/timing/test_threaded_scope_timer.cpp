
#include "phlop/timing/threaded_scope_timer.hpp"

#include <thread>
#include <vector>
#include <cstdlib>
#include <iostream>

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

void fn_mem()
{
    PHLOP_SCOPE("fn_mem", time, mem);

    std::vector<char> v(64 * 1024 * 1024, 1); // value initialized, so resident

    // read back through volatile so the optimizer can't elide the allocation
    char const volatile* const p = v.data();
    std::size_t sum              = 0;
    for (std::size_t i = 0; i < v.size(); i += 4096)
        sum += static_cast<std::size_t>(p[i]);
    if (sum != v.size() / 4096)
        std::abort();
}

void fn3()
{
    PHLOP_SCOPE_TIMER("fn3"); // time only scope, with a mem scope child

    fn_mem();
}

void fn2()
{
    PHLOP_SCOPE_TIMER("fn2");

    std::this_thread::sleep_for(200ms);
    fn0();
}

int main()
{
    std::cout << __FILE__ << std::endl;

    // PHLOP_SCOPE_MEM=0 overrides memory(), see test_scope_timer.py
    phlop::scope_timer().file_name("bin/threaded_scope_timer.txt").memory().init();

    std::thread{[&]() {
        for (std::size_t i = 0; i < 2; ++i)
            fn1();
    }}.join();

    for (std::size_t i = 0; i < 2; ++i)
        fn2();

    std::thread{[&]() { fn3(); }}.join(); // mem scope off the main thread

    {
        PHLOP_SCOPE_TIMER("thread_scope");
        std::thread{[&]() {
            for (std::size_t i = 0; i < 2; ++i)
                fny();
        }}.join();
    }
}
