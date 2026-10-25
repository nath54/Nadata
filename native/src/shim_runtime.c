/*
 * shim_runtime.c — interpreter lifecycle, TLS errors, status mapping.
 *
 * The embedded CPython lives for the process lifetime (deliberately
 * never finalized: engine globals, caches and atexit hooks make a
 * clean teardown fragile, and hosts free the library at exit anyway).
 */

/* Import Modules */
#include "shim_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <windows.h>
#elif defined(__APPLE__) || defined(__unix__)
#  include <dlfcn.h>
#endif

#define BUILTIN_MODULE "builtins"
#define ND_MAX_PATH 1024

/* Native path separator. */
#if defined(_WIN32)
#  define ND_PATH_SEP '\\'
#else
#  define ND_PATH_SEP '/'
#endif

/* ------------------------------------------------------------------ */
/* Interpreter state                                                   */
/* ------------------------------------------------------------------ */

static int g_initialized = 0;       /* interpreter + imports ready    */
static PyObject* g_nadata_cls = NULL;   /* strong: nadata.Nadata      */
static PyObject* g_entity_id_cls = NULL;/* strong: EntityId           */
static PyObject* g_scope_cls = NULL;    /* strong: ExecutionScope     */
static PyObject* g_parser_cls = NULL;   /* strong: ExpressionParser   */
static PyObject* g_queryset_cls = NULL; /* strong: QuerySet           */

/* Set at boot when the bundled (Phase 1b) runtime layout is detected:
 * the directory containing the staged `nadata/` package + `runtime/`. */
static char g_embedded_libdir[ND_MAX_PATH] = { 0 };

/* ------------------------------------------------------------------ */
/* Embedded runtime layout (Phase 1b)                                  */
/* ------------------------------------------------------------------ */

/* 1 when `path` names an existing regular file. */
static int file_exists(const char* path)
{
    FILE* handle = fopen(path, "rb");

    if (handle == NULL) {
        return 0;
    }

    fclose(handle);

    return 1;
}

/* 1 when `path` names an existing directory. */
static int directory_exists(const char* path)
{
#if defined(_WIN32)
    DWORD attributes = GetFileAttributesA(path);

    return attributes != INVALID_FILE_ATTRIBUTES
        && (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
    /* fopen on a directory succeeds on POSIX (reading fails) — the
     * existence probe only. */
    FILE* handle = fopen(path, "rb");

    if (handle != NULL) {
        fclose(handle);
        return 1;
    }

    return 0;
#endif
}

/*
 * Fills `out` with the directory containing THIS library (the shim),
 * so the bundled runtime is found relative to the installation rather
 * than the cwd or env vars. Returns 1 on success.
 */
static int library_directory(char* out, size_t cap)
{
#if defined(_WIN32)
    HMODULE handle = NULL;

    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                            | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR) &nd_ensure_interpreter, &handle)) {
        return 0;
    }

    WCHAR wide[ND_MAX_PATH];

    if (GetModuleFileNameW(handle, wide, ND_MAX_PATH) == 0) {
        return 0;
    }

    if (WideCharToMultiByte(CP_UTF8, 0, wide, -1, out, (int) cap,
                            NULL, NULL) <= 0) {
        return 0;
    }

    char* cut = strrchr(out, '\\');
    char* slash = strrchr(out, '/');

    if (slash > cut) {
        cut = slash;
    }

    if (cut == NULL) {
        return 0;
    }

    *cut = '\0';

    return 1;
#elif defined(__APPLE__) || defined(__unix__)
    Dl_info info;

    if (dladdr((void*) &nd_ensure_interpreter, &info) == 0
        || info.dli_fname == NULL) {
        return 0;
    }

    if (strlen(info.dli_fname) >= cap) {
        return 0;
    }

    strcpy(out, info.dli_fname);

    char* cut = strrchr(out, '/');

    if (cut == NULL) {
        return 0;
    }

    *cut = '\0';

    return 1;
#else
    (void) out;
    (void) cap;

    return 0;
#endif
}

/*
 * Boots the interpreter exactly once per process, choosing the right
 * startup for the environment:
 *
 *   1. Host-provided runtime (Py_IsInitialized): nothing to do — the
 *      ctypes / extension-module case.
 *   2. Bundled layout (Phase 1b): `runtime/` stdlib + `nadata/`
 *      package next to the library -> PyConfig.home points there;
 *      no env vars, no registry, no distro Python.
 *   3. Plain Py_Initialize: dev builds / bare hosts, where sys.path
 *      injection (NADATA_HOME or the embedded libdir) happens later.
 */
static int boot_interpreter(void)
{
    if (Py_IsInitialized()) {
        return 0;
    }

    const char* dev_home = getenv("NADATA_HOME");

    if (dev_home == NULL || dev_home[0] == '\0') {
        char libdir[ND_MAX_PATH];
        char runtime[ND_MAX_PATH];
        char marker[ND_MAX_PATH];

        if (library_directory(libdir, sizeof(libdir))
            && snprintf(marker, sizeof(marker), "%s%cnadata%c__init__.py",
                        libdir, ND_PATH_SEP, ND_PATH_SEP)
               < (int) sizeof(marker)
            && file_exists(marker)
            && snprintf(runtime, sizeof(runtime), "%s%cruntime",
                        libdir, ND_PATH_SEP) < (int) sizeof(runtime)
            && directory_exists(runtime)) {
            /* Bundled runtime found: home it, remember the libdir. */
            PyConfig config;

            PyConfig_InitPythonConfig(&config);
            config.home = Py_DecodeLocale(runtime, NULL);

            PyStatus status = Py_InitializeFromConfig(&config);

            /* PyConfig_Clear frees every string field, including the
             * one we decoded — no manual PyMem_RawFree here. */
            PyConfig_Clear(&config);

            if (PyStatus_Exception(status)) {
                return -1;
            }

            strncpy(g_embedded_libdir, libdir, sizeof(g_embedded_libdir) - 1);

            return 0;
        }
    }

    /* Dev build or bare host: plain initialization. */
    Py_Initialize();

    return 0;
}

/* One cached engine class lookup. Returns new ref or NULL + error. */
static PyObject* cache_class(PyObject** slot, const char* module_name,
                             const char* class_name)
{
    if (*slot != NULL) {
        Py_INCREF(*slot);
        return *slot;
    }

    PyObject* module = PyImport_ImportModule(module_name);

    if (module == NULL) {
        return NULL;
    }

    *slot = PyObject_GetAttrString(module, class_name);
    Py_DECREF(module);

    if (*slot == NULL) {
        return NULL;
    }

    Py_INCREF(*slot);
    return *slot;
}

PyGILState_STATE nd_attach(void)
{
    /*
     * Py_Initialize needs no GIL (it creates it); PyGILState_Ensure
     * requires an initialized runtime, hence the boot first. The boot
     * is idempotent and picks the embedded layout when bundled.
     */
    if (!Py_IsInitialized() && boot_interpreter() != 0) {
        fputs("nadata: failed to boot the embedded CPython runtime\n",
              stderr);
    }

    return PyGILState_Ensure();
}

int nd_ensure_interpreter(void)
{
    if (g_initialized) {
        return 0;
    }

    if (boot_interpreter() != 0) {
        nd_set_error("failed to initialize the embedded CPython runtime");
        return -1;
    }

    /*
     * sys.path injection, in priority order:
     *   1. NADATA_HOME (dev mode) — a checkout or venv site-packages.
     *   2. The embedded libdir — where the staged nadata/ package
     *      lives next to the library (Phase 1b builds).
     */
    const char* home = getenv("NADATA_HOME");
    const char* inject = NULL;

    if (home != NULL && home[0] != '\0') {
        inject = home;
    } else if (g_embedded_libdir[0] != '\0') {
        inject = g_embedded_libdir;
    }

    if (inject != NULL) {
        PyObject* path = PySys_GetObject("path");  /* borrowed */

        if (path == NULL || !PyList_Check(path)) {
            nd_set_error("sys.path unavailable; interpreter broken");
            return -1;
        }

        PyObject* entry = PyUnicode_FromString(inject);

        if (entry == NULL) {
            nd_set_error("path injection source is not valid UTF-8/Unicode");
            return -1;
        }

        if (PyList_Insert(path, 0, entry) != 0) {
            Py_DECREF(entry);
            nd_set_error("failed to prepend the nadata package to sys.path");
            return -1;
        }

        Py_DECREF(entry);
    }

    /* Import the facade class first: it validates the installation. */
    /*
     * faulthandler first: any later native crash prints the Python
     * stack instead of dying silently (invaluable for shim debugging).
     */
    PyObject* fault_mod = PyImport_ImportModule("faulthandler");

    if (fault_mod != NULL) {
        PyObject* enable = nd_attr(fault_mod, "enable");

        if (enable != NULL) {
            PyObject* none = nd_call(enable, NULL, NULL);

            if (none == NULL) {
                PyErr_Clear();
            }

            Py_DECREF(enable);
        }

        Py_DECREF(fault_mod);
    }

    PyErr_Clear();

    PyObject* nadata_module = PyImport_ImportModule("nadata");

    if (nadata_module == NULL) {
        nd_take_pyerror();
        nd_set_error("%s",
                     "cannot import the 'nadata' package into the embedded "
                     "interpreter (set NADATA_HOME to a checkout or venv "
                     "site-packages, or ship the staged nadata/ package "
                     "next to the library)");
        return -1;
    }

    g_nadata_cls = PyObject_GetAttrString(nadata_module, "Nadata");
    Py_DECREF(nadata_module);

    if (g_nadata_cls == NULL) {
        nd_take_pyerror();
        return -1;
    }

    /* Cache every engine class the shim needs (fail fast, once). */
    struct { PyObject** slot; const char* module; const char* name; } needed[] = {
        { &g_entity_id_cls, "nadata.core.database.entity_id", "EntityId" },
        { &g_scope_cls, "nadata.core.expressions.scope", "ExecutionScope" },
        { &g_parser_cls, "nadata.parser.expressions.expression_parser",
          "ExpressionParser" },
        { &g_queryset_cls, "nadata.core.database.queryset", "QuerySet" },
    };

    for (size_t i = 0; i < sizeof(needed) / sizeof(needed[0]); i++) {
        PyObject* cls = cache_class(needed[i].slot, needed[i].module,
                                    needed[i].name);

        if (cls == NULL) {
            nd_take_pyerror();
            return -1;
        }

        Py_DECREF(cls);  /* slot keeps the strong reference */
    }

    g_initialized = 1;

    return 0;
}

PyObject* nd_nadata_cls(void)
{
    return g_nadata_cls;  /* borrowed: interpreter-lifetime strong ref */
}

PyObject* nd_engine_class(const char* module_name, const char* class_name)
{
    if (strcmp(module_name, "nadata.core.database.entity_id") == 0) {
        return cache_class(&g_entity_id_cls, module_name, class_name);
    }
    if (strcmp(module_name, "nadata.core.expressions.scope") == 0) {
        return cache_class(&g_scope_cls, module_name, class_name);
    }
    if (strcmp(module_name, "nadata.parser.expressions.expression_parser") == 0) {
        return cache_class(&g_parser_cls, module_name, class_name);
    }
    if (strcmp(module_name, "nadata.core.database.queryset") == 0) {
        return cache_class(&g_queryset_cls, module_name, class_name);
    }

    /* Uncached fallback for any future engine class. */
    PyObject* module = PyImport_ImportModule(module_name);

    if (module == NULL) {
        return NULL;
    }

    PyObject* cls = PyObject_GetAttrString(module, class_name);

    Py_DECREF(module);

    return cls;
}

/* ------------------------------------------------------------------ */
/* Thread-local errors                                                 */
/* ------------------------------------------------------------------ */

static ND_TLS char g_error[ND_ERROR_BUFFER_SIZE] = { 0 };

void nd_clear_error(void)
{
    g_error[0] = '\0';
}

void nd_set_error(const char* fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    vsnprintf(g_error, ND_ERROR_BUFFER_SIZE, fmt, args);
    va_end(args);
}

const char* nd_tls_error(void)
{
    return g_error[0] == '\0' ? NULL : g_error;
}

/* ------------------------------------------------------------------ */
/* Exception -> status mapping (mirrors tests/abi/contract.py)         */
/* ------------------------------------------------------------------ */

/*
 * Resolves an exception TYPE to a status code. Order mirrors the
 * MRO-precise contract: subclasses (PermissionError < OSError) must
 * be matched first. Called AFTER PyErr_Fetch, so it takes the type
 * explicitly — PyErr_Occurred() would be NULL by then.
 */
static NdStatusCode map_exception_type(PyObject* exc_type)
{
    if (exc_type == NULL) {
        return ND_ERR_RUNTIME;
    }

    struct { const char* module; const char* name; NdStatusCode status; }
    classes[] = {
        { BUILTIN_MODULE, "KeyError", ND_ERR_NOTFOUND },
        { BUILTIN_MODULE, "AttributeError", ND_ERR_NOTFOUND },
        { BUILTIN_MODULE, "IndexError", ND_ERR_NOTFOUND },
        { BUILTIN_MODULE, "NameError", ND_ERR_NAME },
        { BUILTIN_MODULE, "ValueError", ND_ERR_VALUE },
        { BUILTIN_MODULE, "TypeError", ND_ERR_TYPE },
        { BUILTIN_MODULE, "SyntaxError", ND_ERR_SYNTAX },
        { BUILTIN_MODULE, "PermissionError", ND_ERR_PERMISSION },
        { BUILTIN_MODULE, "OSError", ND_ERR_IO },
        { "nadata.core.expressions.scope", "LoopLimitError", ND_ERR_LIMIT },
        { BUILTIN_MODULE, "ZeroDivisionError", ND_ERR_RUNTIME },
        { BUILTIN_MODULE, "OverflowError", ND_ERR_RUNTIME },
        { BUILTIN_MODULE, "RuntimeError", ND_ERR_RUNTIME },
        { BUILTIN_MODULE, "MemoryError", ND_ERR_MEMORY },
    };

    for (size_t i = 0; i < sizeof(classes) / sizeof(classes[0]); i++) {
        PyObject* cls;

        if (strcmp(classes[i].module, BUILTIN_MODULE) == 0) {
            cls = PyObject_GetAttrString(PyImport_AddModule(BUILTIN_MODULE),
                                         classes[i].name);
        } else {
            PyObject* module = PyImport_ImportModule(classes[i].module);

            if (module == NULL) {
                continue;
            }

            cls = PyObject_GetAttrString(module, classes[i].name);
            Py_DECREF(module);
        }

        if (cls == NULL) {
            PyErr_Clear();
            continue;
        }

        int matches = PyErr_GivenExceptionMatches(exc_type, cls);

        Py_DECREF(cls);

        if (matches) {
            return classes[i].status;
        }
    }

    return ND_ERR_RUNTIME;
}

/*
 * Consumes the active exception: maps it to a status, stores the
 * message in TLS, clears the error state. Returns the status so call
 * sites can `return nd_take_pyerror();`.
 */
NdStatusCode nd_take_pyerror(void)
{
    PyObject* type = NULL;
    PyObject* value = NULL;
    PyObject* traceback = NULL;

    if (!PyErr_Occurred()) {
        nd_set_error("internal: nd_take_pyerror with no active exception");
        return ND_ERR_RUNTIME;
    }

    PyErr_Fetch(&type, &value, &traceback);

    NdStatusCode status = map_exception_type(type);

    /* Message: str(exception), or the type name when str() fails. */
    const char* text = NULL;
    PyObject* message = NULL;

    if (value != NULL) {
        message = PyObject_Str(value);
    }

    if (message != NULL) {
        text = PyUnicode_AsUTF8(message);
    }

    if (text != NULL) {
        nd_set_error("%s", text);
    } else {
        const char* name = PyExceptionClass_Name(type);

        nd_set_error("exception of type %s",
                     name != NULL ? name : "<unknown>");
    }

    Py_XDECREF(message);
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);

    return status;
}

/* ------------------------------------------------------------------ */
/* Runtime info (nd_runtime_info lives here: it is interpreter state)  */
/* ------------------------------------------------------------------ */

ND_API char* nd_runtime_info(void)
{
    ND_SHIM_BEGIN();

    /* Interpreter must be up: PyGILState_Ensure does not init Python. */
    if (nd_ensure_interpreter() != 0) {
        ND_SHIM_RETURN(NULL);
    }

#ifdef Py_GIL_DISABLED
    const char* gil_state = "true";
#else
    const char* gil_state = "false";
#endif

    /*
     * Py_GetVersion() appends build details after the version number;
     * take the first space-delimited token so the JSON stays clean.
     */
    char version_short[32];

    if (sscanf(Py_GetVersion(), "%31s", version_short) != 1) {
        version_short[0] = '\0';
    }

    char buffer[256];

#ifdef NADATA_EMBEDDED
    const char* build_kind = "embedded-pbs";
#else
    const char* build_kind = "shim-dev";
#endif

    snprintf(buffer, sizeof(buffer),
             "{\"abi\":%d,\"cpython\":\"%s\",\"free_threaded\":%s,"
             "\"build\":\"%s\"}",
             ND_ABI_VERSION, version_short, gil_state, build_kind);

    char* json = nd_strdup(buffer);

    if (json == NULL) {
        nd_set_error("out of memory composing runtime info");
        ND_SHIM_RETURN(NULL);
    }

    ND_SHIM_RETURN(json);
}
