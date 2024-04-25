
#include "phlop/timing/scope_timer.hpp"

void fny()
{
    PHLOP_SCOPE_TIMER("fny");
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(100ms);
}

void fn0()
{
    PHLOP_SCOPE_TIMER("fn0");
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(100ms);

    fny();

    PHLOP_SCOPE_TIMER("fn0b");
}

void fn1()
{
    PHLOP_SCOPE_TIMER("fn1");
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(100ms);
    PHLOP_SCOPE_TIMER("fn1b");
    fn0();
}

void fn2()
{
    PHLOP_SCOPE_TIMER("fn2");
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(100ms);
    fn0();
}

int main()
{
    // ONLY SUPPORTS SERIAL OPERATIONS!
    phlop::ScopeTimerMan::INSTANCE().file_name("scope_timer.txt").init();

    for (std::size_t i = 0; i < 2; ++i)
        fn1();
    for (std::size_t i = 0; i < 2; ++i)
        fn2();
    for (std::size_t i = 0; i < 2; ++i)
        fny();

    phlop::ScopeTimerMan::INSTANCE().shutdown();
}
