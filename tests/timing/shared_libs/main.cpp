// tests/timing/shared_libs/main.cpp
// loads the core and module libraries RTLD_LOCAL, as python loads extensions,
// and checks they share phlop's state. See sh/test_shared_libs.sh

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>

template<typename F>
F sym(void* lib, char const* name)
{
    auto f = reinterpret_cast<F>(dlsym(lib, name));
    if (!f)
    {
        std::fprintf(stderr, "missing symbol %s\n", name);
        std::exit(1);
    }
    return f;
}

int main(int argc, char** argv)
{
    if (argc < 3)
        return 1;
    void* core   = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    void* module = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!core or !module)
    {
        std::fprintf(stderr, "dlopen: %s\n", dlerror());
        return 1;
    }
    using ptr_fn = void const* (*)();

    sym<void (*)()>(core, "core_init")();

    int failures     = 0;
    auto const check = [&](char const* what, void const* a, void const* b) {
        std::printf("%-26s core=%p module=%p %s\n", what, a, b, a == b ? "shared" : "SEPARATE");
        failures += a != b;
    };
    check("perf::Config::INSTANCE()", sym<ptr_fn>(core, "core_config")(),
          sym<ptr_fn>(module, "module_config")());
    check("ScopeTimerMan::local()", sym<ptr_fn>(core, "core_man_local")(),
          sym<ptr_fn>(module, "module_man_local")());
    std::printf("module sees perf counters: %d\n", sym<bool (*)()>(module, "module_has_perf")());

    auto const work = sym<void (*)()>(module, "module_work");
    work();                                              // a root scope
    sym<void (*)(void (*)())>(core, "core_outer")(work); // nested across libraries

    if (!std::getenv("PHLOP_SHARED_LIBS_NO_SHUTDOWN")) // else exit time destruction order
        sym<void (*)()>(core, "core_shutdown")();
    return failures;
}
