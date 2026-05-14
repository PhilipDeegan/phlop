
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

    phlop::scope_timer().file_name("bin/" + phlop::mpi::scope_timer_file_namer()).init();

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

    MPI_Finalize();
}
