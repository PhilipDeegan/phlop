#ifndef _PHLOP_OS_IO_HPP_
#define _PHLOP_OS_IO_HPP_

// the calling thread's I/O accounting, /proc/thread-self/io (linux >= 3.17),
//  readable by the process itself without privileges. Elsewhere unsupported:
//  read() returns false
//
//  rchar/wchar: bytes through read/write like syscalls, page cache included
//  syscr/syscw: number of those syscalls
//  read_bytes/write_bytes: bytes the block layer fetched/was sent, local
//   filesystems: network filesystem clients (lustre, gpfs, nfs) may not count
//  cancelled_write_bytes: dirty page cache truncated before writeback

#include <array>
#include <cstdint>

#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace phlop::os
{

struct IoCounts
{
    std::uint64_t rchar = 0, wchar = 0, syscr = 0, syscw = 0;
    std::uint64_t read_bytes = 0, write_bytes = 0, cancelled_write_bytes = 0;

    IoCounts operator-(IoCounts const& o) const
    {
        return {rchar - o.rchar,
                wchar - o.wchar,
                syscr - o.syscr,
                syscw - o.syscw,
                read_bytes - o.read_bytes,
                write_bytes - o.write_bytes,
                cancelled_write_bytes - o.cancelled_write_bytes};
    }
};

// the calling thread's /proc/thread-self/io, kept open, re-read with pread
class ThreadIo
{
public:
    static ThreadIo& local()
    {
        static thread_local ThreadIo io;
        return io;
    }

    ~ThreadIo()
    {
#if defined(__linux__)
        if (fd >= 0)
            ::close(fd);
#endif
    }

    // into out, false if unsupported. self_bytes: what this read itself
    //  returned, the kernel adds it (and one syscr) to the counts after the
    //  read, so a later sample sees it: subtract it from the difference
    bool read(IoCounts& out, std::uint64_t& self_bytes)
    {
#if defined(__linux__)
        if (fd >= 0 and pid != ::getpid()) // forked: the fd is the parent's thread
        {
            ::close(fd);
            fd = -1;
        }
        if (fd == -1)
        {
            fd  = ::open("/proc/thread-self/io", O_RDONLY | O_CLOEXEC);
            pid = ::getpid();
        }
        if (fd < 0)
        {
            bool const transient = errno == EMFILE or errno == ENFILE or errno == EINTR
                                   or errno == ENOMEM or errno == EAGAIN;
            fd = transient ? -1 : -2; // retry later, or don't
            return false;
        }
        std::array<char, 512> buf;
        auto const n = ::pread(fd, buf.data(), buf.size(), 0);
        if (n <= 0)
            return false;
        // "name: value" lines in a fixed order, the values in IoCounts order
        std::array<std::uint64_t*, 7> fields{&out.rchar,
                                             &out.wchar,
                                             &out.syscr,
                                             &out.syscw,
                                             &out.read_bytes,
                                             &out.write_bytes,
                                             &out.cancelled_write_bytes};
        std::size_t f = 0;
        for (ssize_t i = 0; i < n and f < fields.size(); ++i)
            if (buf[i] == ':')
            {
                std::uint64_t v = 0;
                ++i;
                while (i < n and buf[i] == ' ')
                    ++i;
                for (; i < n and buf[i] >= '0' and buf[i] <= '9'; ++i)
                    v = v * 10 + static_cast<std::uint64_t>(buf[i] - '0');
                *fields[f++] = v;
            }
        self_bytes = static_cast<std::uint64_t>(n);
        return f == fields.size();
#else
        (void)out;
        (void)self_bytes;
        return false;
#endif
    }

#if defined(__linux__) // only read there, an unused private field elsewhere
private:
    int fd    = -1; // -1 not opened yet, -2 unsupported
    pid_t pid = 0;  // the process fd was opened in
#endif
};

} // namespace phlop::os

#endif /*_PHLOP_OS_IO_HPP_*/
