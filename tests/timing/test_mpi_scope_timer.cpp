
#include "phlop/timing/mpi_scope_timer.hpp"

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

void fn2()
{
    PHLOP_SCOPE_TIMER("fn2");

    std::this_thread::sleep_for(200ms);
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

// static init runs on the main thread
static std::thread::id const main_thread_id = std::this_thread::get_id();

void fn3()
{
    if (std::this_thread::get_id() == main_thread_id)
        std::abort(); // must run on a worker thread, see main

    PHLOP_SCOPE_TIMER("fn3"); // time only scope, with a mem scope child

    fn_mem();
}

int main(int argc, char** argv)
{
    std::cout << __FILE__ << std::endl;

    MPI_Init(&argc, &argv);

    phlop::scope_timer().file_name("bin/" + phlop::mpi::scope_timer_file_namer()).memory().init();

    std::thread{[&]() {
        for (std::size_t i = 0; i < 2; ++i)
            fn1();
    }}.join();

    for (std::size_t i = 0; i < 2; ++i)
        fn2();

    {
        PHLOP_SCOPE_TIMER("thread_scope");
        std::thread{[&]() {
            for (std::size_t i = 0; i < 2; ++i)
                fny();
        }}.join();
    }

    std::thread{[&]() { fn3(); }}.join(); // mem scope off the main thread

    MPI_Finalize();
}
