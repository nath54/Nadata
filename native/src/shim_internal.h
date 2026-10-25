/*
 * shim_internal.h — internal declarations for the nadata C shim.
 *
 * Not installed, not part of the ABI: consumers only see nadata.h.
 * Split of responsibilities:
 *
 *   shim_runtime.c — interpreter lifecycle, thread-local errors,
 *                    exception -> status mapping, runtime info
 *   shim_value.c   — NdValue <-> PyObject conversion, memory ownership
 *   shim_db.c      — lifecycle, introspection, instances, persistence,
 *                    execution (everything on the facade)
 *   shim_query.c   — the QuerySet family
 *
 * Conventions:
 *   - Every public entry begins with ND_SHIM_BEGIN() (attaches the
 *     thread to the interpreter and clears the TLS error) and returns
 *     through ND_SHIM_RETURN(...) (detaches, returns).
 *   - Shim-level failures (NULL args, allocation) call nd_set_error()
 *     and return their status directly.
 *   - Python failures call nd_take_pyerror(), which maps the active
 *     exception to a status AND stores its message in TLS.
 */

#ifndef ND_SHIM_INTERNAL_H
#define ND_SHIM_INTERNAL_H

#include "nadata.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stddef.h>

/* ------------------------------------------------------------------ */
/* Thread-local error storage                                          */
/* ------------------------------------------------------------------ */

#if defined(_MSC_VER)
#  define ND_TLS __declspec(thread)
#else
#  define ND_TLS _Thread_local
#endif

#define ND_ERROR_BUFFER_SIZE 1024

/* GIL guard pair for public entry points. nd_attach() raw-initializes
 * the interpreter when needed (PyGILState_Ensure does NOT init Python
 * and segfaults on an uninitialized runtime), then attaches the
 * thread. Engine imports happen afterwards inside nd_ensure_interpreter. */
#define ND_SHIM_BEGIN()                                  \
    PyGILState_STATE _nd_gil = nd_attach();              \
    nd_clear_error()

#define ND_SHIM_RETURN(status)                           \
    do {                                                 \
        PyGILState_Release(_nd_gil);                     \
        return (status);                                 \
    } while (0)

/* Void variant for functions that release handles (e.g. nd_close). */
#define ND_SHIM_RETURN_VOID()                            \
    do {                                                 \
        PyGILState_Release(_nd_gil);                     \
        return;                                          \
    } while (0)

/* ------------------------------------------------------------------ */
/* Handles (opaque in nadata.h)                                        */
/* ------------------------------------------------------------------ */

struct NdDb {
    PyObject* facade;   /* strong ref to a nadata.Nadata instance */
};

struct NdQuery {
    PyObject* query_set;    /* strong ref to a queryset.QuerySet */
};

/* ------------------------------------------------------------------ */
/* Runtime (shim_runtime.c)                                            */
/* ------------------------------------------------------------------ */

/* Initializes CPython + imports the engine once per process. 0 = ok. */
int nd_ensure_interpreter(void);

/* Raw-initializes the interpreter if needed, then attaches the thread. */
PyGILState_STATE nd_attach(void);

/* Borrowed ref to the nadata.Nadata class (interpreter must be up). */
PyObject* nd_nadata_cls(void);

/* New reference to a cached engine class, or NULL with exception set. */
PyObject* nd_engine_class(const char* module_name, const char* class_name);

/* TLS error buffer management. */
void nd_clear_error(void);
void nd_set_error(const char* fmt, ...);
const char* nd_tls_error(void);

/*
 * Maps the ACTIVE Python exception to a status code, stores its
 * message in TLS, clears the error state, and returns the status.
 */
NdStatusCode nd_take_pyerror(void);

/* ------------------------------------------------------------------ */
/* Small Python helpers                                                */
/* ------------------------------------------------------------------ */

/* New reference to obj.name, or NULL with exception set. */
PyObject* nd_attr(PyObject* obj, const char* name);

/*
 * Calls callable(args, kwargs); either may be NULL (empty tuple /
 * no kwargs). Returns a new reference, or NULL with the exception set.
 */
PyObject* nd_call(PyObject* callable, PyObject* args, PyObject* kwargs);

/* New reference to facade.database (the engine's live database). */
PyObject* nd_facade_database(PyObject* facade);

/*
 * New reference to the StructInstance for a 64-bit entity id, using
 * database.get_by_id(). NULL with exception set on failure.
 */
PyObject* nd_instance_from_id(PyObject* database, int64_t id);

/* New reference to instance.entity_id.value (int). NULL + exception. */
PyObject* nd_entity_id_value(PyObject* instance);

/* ------------------------------------------------------------------ */
/* Value conversion (shim_value.c)                                     */
/* ------------------------------------------------------------------ */

/*
 * Converts an INPUT (borrowed) NdValue into a fresh Python object.
 * `database` is the live database (used to resolve ND_INSTANCE ids);
 * may be NULL only when the value contains no instances.
 */
NdStatusCode nd_py_from_value(PyObject* database, const NdValue* value,
                              PyObject** out);

/*
 * Converts an input args array (ND_ARRAY or NULL) and kwargs dict
 * (ND_DICT or NULL) into fresh Python objects for nd_call.
 */
NdStatusCode nd_py_from_args(PyObject* database,
                             const NdValue* args, const NdValue* kwargs,
                             PyObject** out_args, PyObject** out_kwargs);

/*
 * Converts a Python result into an OUTPUT NdValue the caller owns.
 * Strings/arrays/dicts are heap-allocated; free with nd_free_value().
 * Home types (DateTime, Color, ...) map to ND_ERR_TYPE per contract.
 */
NdStatusCode nd_value_from_py(PyObject* obj, NdValue* out);

/* nd_strdup — dies with ND_ERR_MEMORY semantics (returns NULL). */
char* nd_strdup(const char* text);

#endif /* ND_SHIM_INTERNAL_H */
