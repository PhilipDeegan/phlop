#ifndef _PHLOP_TIMING_NVTX_HPP_
#define _PHLOP_TIMING_NVTX_HPP_

// -DPHLOP_NVTX: every PHLOP_SCOPE and PHLOP_SCOPE_TIMER is also an NVTX range,
// in domain "phlop", so scopes show on the Nsight Systems timeline (nsys
// --trace=nvtx) next to the kernels and copies they launch, and ncu can filter
// kernels by them (--nvtx --nvtx-include "phlop@key/"). Push/pop is per
// thread, as scopes are. The range encloses the timer, so it covers the
// timed region. Names are registered once per call site and thread, as keys are.
//
// NVTX v3 is header only: CUDA's include directory, or github.com/NVIDIA/NVTX.
// Link -ldl. Without a tool attached each range costs a few ns.
// Without PHLOP_NVTX nothing is added.

#include "phlop/macros/def/string.hpp"

#if defined(PHLOP_NVTX)

#include <nvtx3/nvToolsExt.h>

#include <string>
#include <string_view>

namespace phlop::nvtx
{

inline nvtxDomainHandle_t domain()
{
    static nvtxDomainHandle_t const d = nvtxDomainCreateA("phlop");
    return d;
}

inline nvtxStringHandle_t name(std::string_view const key)
{
    return nvtxDomainRegisterStringA(domain(), std::string{key}.c_str());
}

struct Range
{
    explicit Range(nvtxStringHandle_t const n)
    {
        nvtxEventAttributes_t a{};
        a.version            = NVTX_VERSION;
        a.size               = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        a.messageType        = NVTX_MESSAGE_TYPE_REGISTERED;
        a.message.registered = n;
        nvtxDomainRangePushEx(domain(), &a);
    }
    ~Range() { nvtxDomainRangePop(domain()); }

    Range(Range const&)            = delete;
    Range& operator=(Range const&) = delete;
};

} // namespace phlop::nvtx

// key: the scope's report's key, already evaluated, so the expression runs once.
//  thread_local: threaded scopes evaluate their key per thread
#define PHLOP_NVTX_RANGE(key)                                                                      \
    static thread_local nvtxStringHandle_t const PHLOP_STR_CAT(_phlop_nvtx_name_, __LINE__)        \
        = phlop::nvtx::name(key);                                                                  \
    phlop::nvtx::Range const PHLOP_STR_CAT(_phlop_nvtx_, __LINE__){                                \
        PHLOP_STR_CAT(_phlop_nvtx_name_, __LINE__)};

#else

#define PHLOP_NVTX_RANGE(key)

#endif // PHLOP_NVTX

#endif /* _PHLOP_TIMING_NVTX_HPP_ */
