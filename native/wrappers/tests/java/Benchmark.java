import nadata.Nadata;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Benchmark.java — speed + memory protocol for the Java/JNA binding.
 *
 * Mirrors native/wrappers/tests/cpp/benchmark.cpp: the SAME workloads
 * with the SAME sizes so numbers are comparable across languages
 * (see native/wrappers/tests/README.md).
 *
 * Output format (parse-friendly, identical to the C++ port):
 *   bench <name>: <ops> ops in <ms> ms => <ops/sec>
 *   mem: start RSS <MB>, peak RSS <MB>, delta <MB>
 *
 * Usage: java Benchmark [path-to-ex0_simple]
 * (run from the repository root; the db is modified in memory only)
 */
public final class Benchmark {

    /* ------------------------------------------------------------------ */
    /* Peak process memory (best effort per platform)                      */
    /* ------------------------------------------------------------------ */

    /** Minimal psapi binding (jna-platform not required). */
    private interface Psapi extends Library {
        Psapi INSTANCE = Native.load("psapi", Psapi.class);

        boolean GetProcessMemoryInfo(Pointer process, Counters counters,
                                     int size);
    }

    private interface Kernel32 extends Library {
        Kernel32 INSTANCE = Native.load("kernel32", Kernel32.class);

        Pointer GetCurrentProcess();
    }

    /** PROCESS_MEMORY_COUNTERS (x64: DWORD + DWORD + 8x SIZE_T). */
    public static class Counters extends Structure {
        public int cb;
        public int PageFaultCount;
        public long PeakWorkingSetSize;
        public long WorkingSetSize;
        public long QuotaPeakPagedPoolUsage;
        public long QuotaPagedPoolUsage;
        public long QuotaPeakNonPagedPoolUsage;
        public long QuotaNonPagedPoolUsage;
        public long PagefileUsage;
        public long PeakPagefileUsage;

        @Override
        protected java.util.List<String> getFieldOrder() {
            return java.util.Arrays.asList("cb", "PageFaultCount",
                "PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage");
        }
    }

    private static long peak_rss_bytes() {
        String os = System.getProperty("os.name", "").toLowerCase();

        if (os.contains("win")) {
            Counters counters = new Counters();
            counters.cb = counters.size();

            if (Psapi.INSTANCE.GetProcessMemoryInfo(
                    Kernel32.INSTANCE.GetCurrentProcess(), counters,
                    counters.size())) {
                return counters.PeakWorkingSetSize;
            }

            return 0;
        }

        if (os.contains("linux")) {
            try {
                for (String line : Files.readAllLines(
                        Path.of("/proc/self/status"))) {
                    if (line.startsWith("VmHWM:")) {
                        return Long.parseLong(
                            line.replaceAll("[^0-9]", "")) * 1024;
                    }
                }
            } catch (Exception ignored) {
                // fall through
            }

            return 0;
        }

        return 0;   /* macOS/other: n/a (documented) */
    }

    private static double mb(long bytes) {
        return bytes / (1024.0 * 1024.0);
    }

    /* ------------------------------------------------------------------ */
    /* Timing + check helpers                                              */
    /* ------------------------------------------------------------------ */

    private static double elapsed_ms(long start_nanos) {
        return (System.nanoTime() - start_nanos) / 1e6;
    }

    private static void check(String name, boolean ok) {
        System.out.printf("check %s: %s%n", name, ok ? "ok" : "FAIL");
    }

    /* ------------------------------------------------------------------ */
    /* Workloads                                                           */
    /* ------------------------------------------------------------------ */

    public static void main(String[] args) throws Exception {
        System.setOut(new java.io.PrintStream(System.out, true, "UTF-8"));

        String lib_path = System.getProperty("nadata.library");

        if (lib_path == null) {
            String os = System.getProperty("os.name", "").toLowerCase();
            String dir = os.contains("win") ? "win32"
                : os.contains("mac") ? "darwin" : "linux";
            String file = os.contains("win") ? "nadata.dll"
                : os.contains("mac") ? "libnadata.dylib" : "libnadata.so";

            lib_path = Path.of("native", "build", dir, file).toString();
        }

        Nadata.loadLibrary(lib_path);

        final String ex0 = args.length > 0 ? args[0] : "./examples/ex0_simple";
        final long start_rss = peak_rss_bytes();

        /* 1. load — 20 full loads. */
        {
            final int ops = 20;
            long start = System.nanoTime();

            for (int i = 0; i < ops; i++) {
                try (Nadata.Database db = Nadata.Database.load(ex0)) {
                    // load + close
                }
            }

            double ms = elapsed_ms(start);
            System.out.printf(
                "bench load: %d ops in %.1f ms => %.0f ops/sec%n",
                ops, ms, ops * 1000.0 / ms);
        }

        try (Nadata.Database db = Nadata.Database.load(ex0)) {
            final long tom = db.getSymbolId("tom");

            /* Verify the fixture before timing anything. */
            check("fixture/birthplace",
                "Paris".equals(db.getAttribute("tom", "birthplace")));

            /* 2. get_attribute. */
            {
                final int ops = 10000;
                long start = System.nanoTime();
                long sink = 0;

                for (int i = 0; i < ops; i++) {
                    sink += ((String) db.getAttribute(
                        "tom", "birthplace")).charAt(0);
                }

                double ms = elapsed_ms(start);
                System.out.printf(
                    "bench get_attribute: %d ops in %.1f ms => %.0f ops/sec"
                        + " (sink %d)%n", ops, ms, ops * 1000.0 / ms, sink);
            }

            /* 3. set_attribute (idempotent write). */
            {
                final int ops = 10000;
                long start = System.nanoTime();

                for (int i = 0; i < ops; i++) {
                    db.setAttribute("tom", "birthplace", "Lyon");
                }

                double ms = elapsed_ms(start);
                System.out.printf(
                    "bench set_attribute: %d ops in %.1f ms => %.0f ops/sec%n",
                    ops, ms, ops * 1000.0 / ms);
            }

            check("set_attribute/roundtrip",
                "Lyon".equals(db.getAttribute("tom", "birthplace")));

            /* 4. query where_str + count (full chain per op). */
            {
                final int ops = 1000;
                long start = System.nanoTime();
                long sink = 0;

                for (int i = 0; i < ops; i++) {
                    try (Nadata.Query q = db.query("Human")) {
                        q.whereStr("h", "h.name == \"Tom\"");
                        sink += q.count();
                    }
                }

                double ms = elapsed_ms(start);
                System.out.printf(
                    "bench query_where_count: %d ops in %.1f ms => %.0f"
                        + " ops/sec (sink %d)%n",
                    ops, ms, ops * 1000.0 / ms, sink);

                check("query_where_count/expected", sink == ops);
            }

            /* 5. create_instance (grows the db). */
            {
                db.executeSchema("class Weapon { name: Str = \"Sword\"; }");

                final int ops = 1000;
                long start = System.nanoTime();
                long sink = 0;

                for (int i = 0; i < ops; i++) {
                    Map<String, Object> attributes = new HashMap<>();
                    attributes.put("name", "Blade " + i);

                    sink += db.createInstance("Weapon", attributes, null);
                }

                double ms = elapsed_ms(start);
                System.out.printf(
                    "bench create_instance: %d ops in %.1f ms => %.0f"
                        + " ops/sec (last id %d)%n",
                    ops, ms, ops * 1000.0 / ms, sink);

                check("create_instance/count",
                    db.getCount("Weapon") == ops);
            }

            /* 6. call_function. */
            {
                final int ops = 10000;
                long start = System.nanoTime();
                int sink = 0;

                for (int i = 0; i < ops; i++) {
                    if (Boolean.TRUE.equals(db.callFunction("is_adult",
                            Nadata.instance(tom)))) {
                        sink++;
                    }
                }

                double ms = elapsed_ms(start);
                System.out.printf(
                    "bench call_function: %d ops in %.1f ms => %.0f"
                        + " ops/sec (sink %d)%n",
                    ops, ms, ops * 1000.0 / ms, sink);

                check("call_function/expected", sink == ops);   // tom is 23
            }
        }

        final long end_rss = peak_rss_bytes();

        System.out.printf(
            "mem: start RSS %.1f MB, peak RSS %.1f MB, delta %.1f MB%n",
            mb(start_rss), mb(end_rss), mb(end_rss - start_rss));
    }
}
