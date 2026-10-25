/*
 * benchmark.cpp — speed + memory protocol for native bindings.
 *
 * The reference implementation of the protocol in
 * native/wrappers/tests/README.md: other languages run the SAME
 * workloads with the SAME sizes so results are comparable.
 *
 * Output format (parse-friendly):
 *   bench <name>: <ops> ops in <ms> ms => <ops/sec>
 *   mem: start RSS <MB>, peak RSS <MB>, delta <MB>
 *
 * Usage: benchmark [path-to-ex0_simple]
 * (run from the repository root; the db is modified in memory only)
 */

/* Import Modules */
#include "nadata.hpp"

#if defined(_WIN32)
#  include <windows.h>
#  include <psapi.h>
#elif defined(__linux__)
#  include <cstdio>
#endif

#include <chrono>
#include <cstdio>
#include <string>

/* ------------------------------------------------------------------ */
/* Peak process memory (best effort per platform)                      */
/* ------------------------------------------------------------------ */

static size_t peak_rss_bytes()
{
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters;

    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters,
                             sizeof(counters))) {
        return counters.PeakWorkingSetSize;
    }

    return 0;
#elif defined(__linux__)
    std::FILE* status = std::fopen("/proc/self/status", "r");

    if (status == nullptr) {
        return 0;
    }

    char line[256];
    size_t peak = 0;

    while (std::fgets(line, sizeof(line), status) != nullptr) {
        if (std::strncmp(line, "VmHWM:", 6) == 0) {
            peak = static_cast<size_t>(std::atoll(line + 6)) * 1024;

            break;
        }
    }

    std::fclose(status);

    return peak;
#else
    return 0;   /* macOS/other: n/a (documented) */
#endif
}

static double mb(size_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

/* ------------------------------------------------------------------ */
/* Timing helpers                                                      */
/* ------------------------------------------------------------------ */

using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static void check(const char* name, int ok)
{
    std::printf("check %s: %s\n", name, ok ? "ok" : "FAIL");
}

/* ------------------------------------------------------------------ */
/* Workloads                                                           */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    const char* ex0 = argc > 1 ? argv[1] : "./examples/ex0_simple";
    const size_t start_rss = peak_rss_bytes();

    /* 1. load — N full loads. */
    {
        const int ops = 20;
        Clock::time_point start = Clock::now();

        for (int i = 0; i < ops; i++) {
            nadata::Database db = nadata::Database::load(ex0);
        }

        double ms = elapsed_ms(start, Clock::now());

        std::printf("bench load: %d ops in %.1f ms => %.0f ops/sec\n",
                    ops, ms, ops * 1000.0 / ms);
    }

    nadata::Database db = nadata::Database::load(ex0);
    const int64_t tom = db.get_symbol_id("tom");

    /* Verify the fixture before timing anything. */
    check("fixture/birthplace",
          std::string(db.get_attribute("tom", "birthplace").as_str())
              == "Paris");

    /* 2. get_attribute. */
    {
        const int ops = 10000;
        Clock::time_point start = Clock::now();
        double sink = 0;

        for (int i = 0; i < ops; i++) {
            sink += db.get_attribute("tom", "birthplace").as_str()[0];
        }

        double ms = elapsed_ms(start, Clock::now());

        std::printf(
            "bench get_attribute: %d ops in %.1f ms => %.0f ops/sec "
            "(sink %.0f)\n", ops, ms, ops * 1000.0 / ms, sink);
    }

    /* 3. set_attribute (idempotent write). */
    {
        const int ops = 10000;
        Clock::time_point start = Clock::now();

        for (int i = 0; i < ops; i++) {
            db.set_attribute("tom", "birthplace",
                             nadata::Value::str("Lyon"));
        }

        double ms = elapsed_ms(start, Clock::now());

        std::printf("bench set_attribute: %d ops in %.1f ms => %.0f ops/sec\n",
                    ops, ms, ops * 1000.0 / ms);
    }

    check("set_attribute/roundtrip",
          std::string(db.get_attribute("tom", "birthplace").as_str())
              == "Lyon");

    /* 4. query where_str + count (full chain per op). */
    {
        const int ops = 1000;
        Clock::time_point start = Clock::now();
        int64_t sink = 0;

        for (int i = 0; i < ops; i++) {
            nadata::Query q = db.query("Human");
            q.where_str("h", "h.name == \"Tom\"");
            sink += q.count();
        }

        double ms = elapsed_ms(start, Clock::now());

        std::printf(
            "bench query_where_count: %d ops in %.1f ms => %.0f ops/sec "
            "(sink %lld)\n", ops, ms, ops * 1000.0 / ms,
            static_cast<long long>(sink));

        check("query_where_count/expected", sink == ops);   // exactly one Tom
    }

    /* 5. create_instance (grows the db). */
    {
        db.execute_schema("class Weapon { name: Str = \"Sword\"; }");

        const int ops = 1000;
        Clock::time_point start = Clock::now();
        int64_t sink = 0;

        for (int i = 0; i < ops; i++) {
            sink += db.create_instance(
                "Weapon",
                {{"name", nadata::Value::str("Blade " + std::to_string(i))}});
        }

        double ms = elapsed_ms(start, Clock::now());

        std::printf(
            "bench create_instance: %d ops in %.1f ms => %.0f ops/sec "
            "(last id %lld)\n", ops, ms, ops * 1000.0 / ms,
            static_cast<long long>(sink));

        check("create_instance/count",
              db.count("Weapon") == ops);   // all rows landed
    }

    /* 6. call_function. */
    {
        const int ops = 10000;
        Clock::time_point start = Clock::now();
        int sink = 0;

        for (int i = 0; i < ops; i++) {
            nadata::Value result = db.call_function(
                "is_adult",
                nadata::Value::array({nadata::Value::instance(tom)}));

            sink += result.as_bool() ? 1 : 0;
        }

        double ms = elapsed_ms(start, Clock::now());

        std::printf(
            "bench call_function: %d ops in %.1f ms => %.0f ops/sec "
            "(sink %d)\n", ops, ms, ops * 1000.0 / ms, sink);

        check("call_function/expected", sink == ops);   // tom (23) is adult
    }

    const size_t end_rss = peak_rss_bytes();

    std::printf("mem: start RSS %.1f MB, peak RSS %.1f MB, delta %.1f MB\n",
                mb(start_rss), mb(end_rss), mb(end_rss - start_rss));

    return 0;
}
