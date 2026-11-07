#ifndef _PHLOP_OS_MEM_HPP_
#define _PHLOP_OS_MEM_HPP_

// process memory queries (bytes, 0 if unsupported)
//  uses mkn.kul's standalone header if available, else a vendored copy of it
//  define PHLOP_NO_KUL to force the vendored copy

#if !defined(PHLOP_NO_KUL) && __has_include("mkn/kul/mem.hpp")
#include "mkn/kul/mem.hpp"
namespace phlop::os
{
namespace mem = mkn::kul::mem;
} // namespace phlop::os
#else
#include "phlop/os/kul_mem.hpp"
namespace phlop::os
{
namespace mem = phlop::os::kul_mem;
} // namespace phlop::os
#endif

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace phlop::os
{

// reset the process peak resident memory (VmHWM, getrusage ru_maxrss) to the
//  current rss, so later peaks are seen even below an earlier high. linux only
//  (/proc/self/clear_refs, >= 4.0), false if unsupported or not permitted
inline bool reset_peak_rss()
{
#if defined(__linux__)
    int const fd = ::open("/proc/self/clear_refs", O_WRONLY | O_CLOEXEC);
    if (fd < 0)
        return false;
    bool const ok = ::write(fd, "5", 1) == 1;
    ::close(fd);
    return ok;
#else
    return false;
#endif
}

} // namespace phlop::os

#endif /*_PHLOP_OS_MEM_HPP_*/
