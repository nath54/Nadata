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

#define BUILTIN_MODULE "builtins"

/* ------------------------------------------------------------------ */
/* Interpreter state                                                   */
/* ------------------------------------------------------------------ */

static int g_initialized = 0;       /* interpreter + imports ready    */
static PyObject* g_nadata_cls = NULL;   /* strong: nadata.Nadata      */
static PyObject* g_entity_id_cls = NULL;/* strong: EntityId           */
static PyObject* g_scope_cls = NULL;    /* strong: ExecutionScope     */
static PyObject* g_parser_cls = NULL;   /* strong: ExpressionParser   */
static PyObject* g_queryset_cls = NULL; /* strong: QuerySet           */

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
     * requires an initialized runtime, hence the guard first.
     */
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    return PyGILState_Ensure();
}

int nd_ensure_interpreter(void)
{
    if (g_initialized) {
        return 0;
    }

    Py_Initialize();

    if (!Py_IsInitialized()) {
        nd_set_error("failed to initialize the embedded CPython runtime");
        return -1;
    }

    /*
     * Optional dev-mode path injection (Phase 1a): NADATA_HOME points
     * at a checkout / venv site-packages containing the nadata package.
     * The embedded PBS builds (Phase 1b) bake this into the runtime
     * layout instead — hosts never need to set it there.
     */
    const char* home = getenv("NADATA_HOME");

    if (home != NULL && home[0] != '\0') {
        PyObject* path = PySys_GetObject("path");  /* borrowed */

        if (path == NULL || !PyList_Check(path)) {
            nd_set_error("sys.path unavailable; interpreter broken");
            return -1;
        }

        PyObject* entry = PyUnicode_FromString(home);

        if (entry == NULL) {
            nd_set_error("NADATA_HOME is not valid UTF-8/Unicode");
            return -1;
        }

        if (PyList_Insert(path, 0, entry) != 0) {
            Py_DECREF(entry);
            nd_set_error("failed to prepend NADATA_HOME to sys.path");
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
                     "site-packages for dev builds)");
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

    snprintf(buffer, sizeof(buffer),
             "{\"abi\":%d,\"cpython\":\"%s\",\"free_threaded\":%s,"
             "\"build\":\"shim-dev\"}",
             ND_ABI_VERSION, version_short, gil_state);

    char* json = nd_strdup(buffer);

    if (json == NULL) {
        nd_set_error("out of memory composing runtime info");
        ND_SHIM_RETURN(NULL);
    }

    ND_SHIM_RETURN(json);
}
