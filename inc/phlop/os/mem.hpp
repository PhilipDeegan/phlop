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

#endif /*_PHLOP_OS_MEM_HPP_*/
