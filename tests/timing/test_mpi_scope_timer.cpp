
#include "phlop/timing/mpi_scope_timer.hpp"

#include <thread>
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

int main(int argc, char** argv)
{
    std::cout << __FILE__ << std::endl;

    MPI_Init(&argc, &argv);

    phlop::mpi::init_scope_timer("mpi_scope_timer.bin");

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

    phlop::threaded::ScopeTimerMan::INSTANCE().shutdown();

    MPI_Finalize();
}
