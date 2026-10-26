package nadata.bundled;

import com.sun.jna.NativeLibrary;
import nadata.Nadata;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Stream;

/**
 * NadataBundled — self-extracting loader for the fat-jar distribution.
 *
 * The nadata-jvm-<version>.jar contains the Java binding classes plus
 * one or more complete staged bundles (native shim + engine package +
 * embedded CPython) under nadata-native/&lt;target&gt;/. On first use
 * this class extracts the matching bundle to ~/.nadata/native/
 * &lt;version&gt;/&lt;target&gt;/ (overridable), preloads the runtime's
 * libpython by absolute path (so no PATH or env var is needed), and
 * loads the shim from there.
 *
 * One-liner for consumers:
 *
 *   NadataBundled.install();
 *   try (Nadata.Database db = Nadata.Database.load("./examples/ex0_simple")) {
 *       ...
 *   }
 *
 * Kotlin consumers can additionally drop NadataExt.kt into their
 * sources for the idiomatic layer.
 */
public final class NadataBundled {

    /** Fallback when nadata-native/version.txt is absent. */
    private static final String FALLBACK_VERSION = "0.1.0";

    private static final String RESOURCE_ROOT = "nadata-native";

    private static String bundleDir;
    private static boolean installed;

    private NadataBundled() {
    }

    /**
     * Extracts (once) and loads the bundle for the running platform.
     *
     * @return the directory the bundle was installed into
     */
    public static synchronized String install() throws IOException {
        if (installed) {
            return bundleDir;
        }

        String target = detectTarget();
        Path dest = Paths.get(System.getProperty("nadata.cache",
                System.getProperty("user.home") + "/.nadata/native"))
            .resolve(version()).resolve(target);

        if (!Files.exists(dest.resolve(".complete"))) {
            extract(target, dest);
            Files.write(dest.resolve(".complete"), new byte[0]);
        }

        preloadRuntime(dest, target);

        Nadata.loadLibrary(dest.resolve(libraryName()).toString());

        installed = true;
        bundleDir = dest.toString();

        return bundleDir;
    }

    /** The directory the bundle was installed into (after install()). */
    public static String bundleDir() {
        if (!installed) {
            throw new IllegalStateException(
                "call NadataBundled.install() first");
        }

        return bundleDir;
    }

    /* ------------------------------------------------------------------ */
    /* Platform + version detection                                        */
    /* ------------------------------------------------------------------ */

    private static String version() {
        try (InputStream stream = NadataBundled.class.getResourceAsStream(
                "/" + RESOURCE_ROOT + "/version.txt")) {
            if (stream != null) {
                return new String(stream.readAllBytes()).trim();
            }
        } catch (IOException ignored) {
            // fall through to the constant
        }

        return FALLBACK_VERSION;
    }

    private static String detectTarget() {
        String os = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch", "").toLowerCase();

        String machine = arch.contains("aarch64") || arch.contains("arm")
            ? "aarch64" : "x86_64";

        if (os.contains("win")) {
            return machine + "-pc-windows-msvc";
        }

        if (os.contains("mac") || os.contains("darwin")) {
            return machine + "-apple-darwin";
        }

        return machine + "-unknown-linux-gnu";
    }

    private static String libraryName() {
        String os = System.getProperty("os.name", "").toLowerCase();

        if (os.contains("win")) {
            return "nadata.dll";
        }

        return os.contains("mac") ? "libnadata.dylib" : "libnadata.so";
    }

    /**
     * Preloads the runtime's libpython by ABSOLUTE path: once the OS
     * loader has it, the shim's own dependency resolves from the
     * already-loaded module on all three platforms — no PATH or
     * LD_LIBRARY_PATH/DYLD_LIBRARY_PATH manipulation needed.
     */
    private static void preloadRuntime(Path dest, String target) {
        String os = System.getProperty("os.name", "").toLowerCase();
        List<Path> candidates = new ArrayList<>();

        if (os.contains("win")) {
            // A copy of the runtime dll sits next to the shim.
            try (Stream<Path> files = Files.list(dest)) {
                files.filter(path -> path.getFileName().toString()
                        .matches("python3\\d+\\.dll"))
                    .forEach(candidates::add);
            } catch (IOException ignored) {
                // fall through
            }
        } else {
            try (Stream<Path> files = Files.list(
                    dest.resolve("runtime").resolve("lib"))) {
                files.filter(path -> path.getFileName().toString()
                        .startsWith("libpython3"))
                    .filter(path -> !path.toString().endsWith(".so"))
                    .forEach(candidates::add);
            } catch (IOException ignored) {
                // fall through
            }
        }

        // Prefer the longest name (libpython3.12.so.1.0 over .so.1).
        candidates.sort((a, b) -> b.toString().length()
            - a.toString().length());

        if (!candidates.isEmpty()) {
            NativeLibrary.getInstance(candidates.get(0).toAbsolutePath()
                .toString());
        }
        // No candidate -> let the shim's own loading produce a clear
        // error rather than guessing here.
    }

    /* ------------------------------------------------------------------ */
    /* Extraction                                                          */
    /* ------------------------------------------------------------------ */

    private static void extract(String target, Path dest) throws IOException {
        Path jarPath = jarLocation();

        if (jarPath != null && Files.isRegularFile(jarPath)) {
            try (JarFile jar = new JarFile(jarPath.toFile())) {
                String prefix = RESOURCE_ROOT + "/" + target + "/";
                List<JarEntry> entries = new ArrayList<>();

                for (JarEntry entry : jar.stream().toArray(JarEntry[]::new)) {
                    if (entry.getName().startsWith(prefix)
                            && !entry.isDirectory()) {
                        entries.add(entry);
                    }
                }

                if (entries.isEmpty()) {
                    throw new IOException("bundle missing from jar: "
                        + prefix + " (built for another platform?)");
                }

                for (JarEntry entry : entries) {
                    Path out = dest.resolve(
                        entry.getName().substring(prefix.length()));

                    Files.createDirectories(out.getParent());
                    Files.copy(jar.getInputStream(entry), out,
                        StandardCopyOption.REPLACE_EXISTING);
                }

                return;
            }
        }

        // Loose classes (dev builds): look for a sibling bundle dir.
        Path source = Paths.get(RESOURCE_ROOT, target);

        if (!Files.isDirectory(source)) {
            throw new IOException("no jar and no bundle directory at "
                + source.toAbsolutePath());
        }

        copyTree(source, dest);
    }

    private static void copyTree(Path source, Path dest) throws IOException {
        try (Stream<Path> walk = Files.walk(source)) {
            for (Path entry : walk.collect(java.util.stream.Collectors
                    .toList())) {
                Path out = dest.resolve(source.relativize(entry).toString());

                if (Files.isDirectory(entry)) {
                    Files.createDirectories(out);
                } else {
                    Files.createDirectories(out.getParent());
                    Files.copy(entry, out,
                        StandardCopyOption.REPLACE_EXISTING);
                }
            }
        }
    }

    private static Path jarLocation() throws IOException {
        try {
            return Path.of(NadataBundled.class.getProtectionDomain()
                .getCodeSource().getLocation().toURI());
        } catch (java.net.URISyntaxException error) {
            throw new IOException("cannot locate the jar", error);
        }
    }
}
