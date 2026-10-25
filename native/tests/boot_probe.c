/*
 * boot_probe.c — minimal embedded-boot probe for the staged runtime:
 * Py_InitializeFromConfig(home=runtime) then a trivial expression.
 * Compiled against the PBS headers, linked against python312.lib.
 */
#define PY_SSIZE_T_CLEAN
#define _CRT_SECURE_NO_WARNINGS
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
    const char* runtime = argc > 1 ? argv[1] : ".";
    int use_home = argc > 2 ? atoi(argv[2]) : 1;

    fprintf(stderr, "probe: start (home=%d)\n", use_home);

    PyConfig config;

    fprintf(stderr, "probe: init config\n");
    PyConfig_InitPythonConfig(&config);

    if (use_home) {
        fprintf(stderr, "probe: decode home=%s\n", runtime);
        config.home = Py_DecodeLocale(runtime, NULL);
    }

    fprintf(stderr, "probe: initializing...\n");

    PyStatus status = Py_InitializeFromConfig(&config);

    /* PyConfig_Clear frees every string field, including the one we
     * decoded — no manual PyMem_RawFree here (double free). */
    PyConfig_Clear(&config);

    if (PyStatus_Exception(status)) {
        fprintf(stderr, "boot FAILED: %s\n",
                status.err_msg ? status.err_msg : "<null>");

        if (status.func) {
            fprintf(stderr, "  in %s\n", status.func);
        }

        return 1;
    }

    fprintf(stderr, "boot ok\n");
    fprintf(stderr, "version: %s\n", Py_GetVersion());

    int rc = PyRun_SimpleString(
        "import sys; print('exe:', sys.executable)\n"
        "print('path:', sys.path[:3])\n"
        "import zlib; print('zlib ok')\n"
        "print('boot probe complete')\n");

    fprintf(stderr, "PyRun rc=%d\n", rc);

    Py_Finalize();

    return rc;
}
