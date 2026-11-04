#ifndef _PHLOP_TIMING_SCOPE_HPP_
#define _PHLOP_TIMING_SCOPE_HPP_

// PHLOP_SCOPE("key", time, perf_light) building blocks, shared by the scope,
// threaded and mpi timers. Options are phlop::scope_key names, no `using`
// needed: PHLOP_SCOPE_FLAGS brings them into scope for its own expression
// only, so a bare `time` does not clash with ::time. No options means
// PHLOP_SCOPE_DEFAULT, eg -DPHLOP_SCOPE_DEFAULT="time, perf_light" to count
// in every bare PHLOP_SCOPE. Scopes always record time, perf_* add counters
// when PHLOP_PERF_LIGHT/ALL are set, see perf_counters.hpp. mem adds resident
// memory on entry/exit when ScopeTimerMan::memory() or PHLOP_SCOPE_MEM=1

#include "phlop/timing/common_timer.hpp"
#include "phlop/timing/perf_counters.hpp"

#include <cstdint>
#include <type_traits>

// msvc ignores [[no_unique_address]], a time only Scope must stay a plain Timer
#if defined(_MSC_VER)
#define PHLOP_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define PHLOP_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

#if !defined(PHLOP_SCOPE_DEFAULT)
#define PHLOP_SCOPE_DEFAULT time
#endif

#define PHLOP_SCOPE_FLAGS(...)                                                                     \
    [] {                                                                                           \
        using enum phlop::scope_key;                                                               \
        return phlop::scope_flags(__VA_ARGS__);                                                    \
    }()

namespace phlop
{

enum class scope_key : std::uint32_t {
    time       = 1u << 0,
    perf_light = 1u << 1, // PHLOP_PERF_LIGHT events, eg cycles, instructions, RAM fetches
    perf_all   = 1u << 2, // PHLOP_PERF_ALL events, eg a topdown set
    mem        = 1u << 3, // resident memory (rss, process peak) on entry/exit
};

template<typename... Keys>
constexpr std::uint32_t scope_flags(Keys const... keys)
{
    if constexpr (sizeof...(Keys) == 0)
    {
        using enum scope_key;
        return scope_flags(PHLOP_SCOPE_DEFAULT);
    }
    else
        return (static_cast<std::uint32_t>(keys) | ...);
}

// perf level written to the timer file: 2 perf_all, 1 perf_light, 0 none
constexpr std::uint8_t scope_perf_level(std::uint32_t const flags)
{
    return flags & static_cast<std::uint32_t>(scope_key::perf_all)     ? 2
           : flags & static_cast<std::uint32_t>(scope_key::perf_light) ? 1
                                                                       : 0;
}


// A timer scope, plus the calling thread's counter group around it when F asks
// for a perf level. Without perf/mem flags it is exactly a Timer. Timer needs
// pending_level / pending_counts / pending_mem(_in), moved into its snapshot
// as it destructs, and Man::INSTANCE().active/_memory. Counter reads sit
// inside the timed region, memory samples outside it (~10us each)
template<std::uint32_t F, typename Timer, typename Man>
struct ScopeT
{
    static constexpr std::uint8_t level = scope_perf_level(F);
    static constexpr bool has_mem       = (F & static_cast<std::uint32_t>(scope_key::mem)) != 0;

    template<typename Report>
    ScopeT(Report& r)
        : timer{r}
    {
        if constexpr (level > 0)
        {
            auto& config     = perf::Config::INSTANCE();
            state.generation = config.generation.load(std::memory_order_relaxed);
            state.on = Man::INSTANCE().active and config.has(level) and perf::local().ensure()
                       and perf::local().read(state.begin);
        }
    }

    ~ScopeT()
    {
        if constexpr (has_mem)
            if (mem.on) // exit sample is taken by the timer, after its time
            {
                timer.pending_mem    = true;
                timer.pending_mem_in = mem.in;
            }
        if constexpr (level > 0)
        {
            perf::Counts end;
            // a counter re-init inside the scope (init/reset) restarted the
            // group, begin is from the old one: drop the counts, not wrap them
            auto const same = state.generation
                              == perf::Config::INSTANCE().generation.load(std::memory_order_relaxed);
            if (state.on and same and perf::local().read(end))
            {
                auto const& idx = perf::Config::INSTANCE().levels[level].idx;
                timer.pending_counts.resize(idx.size());
                for (std::size_t i = 0; i < idx.size(); ++i)
                    timer.pending_counts[i] = end[idx[i]] - state.begin[idx[i]];
                timer.pending_level = level;
            }
        }
    }

    struct Counting
    {
        perf::Counts begin;
        std::uint64_t generation = 0;
        bool on                  = false;
    };
    struct NoCounting
    {
    };

    struct Sampling
    {
        MemSample in{};
        bool on = false;
    };
    struct NoSampling // distinct from NoCounting, so both can share no storage
    {
    };

    static auto mem_begin()
    {
        if constexpr (has_mem)
        {
            auto const& man = Man::INSTANCE();
            Sampling m;
            m.on = man.active and man._memory.load(std::memory_order_relaxed);
            if (m.on)
                m.in = MemSample::now();
            return m;
        }
        else
            return NoSampling{};
    }

    // before timer: initialised first, so the entry sample is not timed
    PHLOP_NO_UNIQUE_ADDRESS std::conditional_t<has_mem, Sampling, NoSampling> mem = mem_begin();
    Timer timer;
    PHLOP_NO_UNIQUE_ADDRESS std::conditional_t<(level > 0), Counting, NoCounting> state;
};

} // namespace phlop

#endif /*_PHLOP_TIMING_SCOPE_HPP_*/
