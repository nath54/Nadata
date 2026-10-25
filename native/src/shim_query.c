/*
 * shim_query.c — the QuerySet family of the ABI.
 *
 * Nadata's QuerySet methods return NEW QuerySet objects; this shim
 * keeps ONE handle (struct NdQuery) and rebinds it in place, so
 * nd_query_order_by(...) mutates *qs — exactly as documented in
 * native/nadata.h.
 *
 * nd_query_where_str evaluates the predicate text with the engine's
 * own parser (the universal filter path); nd_query_where_cb filters a
 * native callback over qs.instances and rebuilds the QuerySet.
 */

/* Import Modules */
#include "shim_internal.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Internal helpers                                                    */
/* ------------------------------------------------------------------ */

/* Calls a QuerySet method by name; new ref or NULL with exc set. */
static PyObject* call_query(PyObject* query_set, const char* method,
                            PyObject* args, PyObject* kwargs)
{

    PyObject* callable = nd_attr(query_set, method);

    if (callable == NULL) {
        return NULL;
    }

    PyObject* result = nd_call(callable, args, kwargs);

    Py_DECREF(callable);

    return result;
}

/* Replaces the handle's bound QuerySet (steals the new reference). */
static void rebind(NdQuery* qs, PyObject* new_query_set)
{
    Py_SETREF(qs->query_set, new_query_set);
}

/* The live database behind a query handle (new ref) or NULL + error. */
static PyObject* query_database(PyObject* query_set)
{
    PyObject* database = nd_attr(query_set, "database");

    if (database == NULL) {
        return NULL;
    }

    if (database == Py_None) {
        Py_DECREF(database);
        nd_set_error("query set has no bound database");
        return NULL;
    }

    return database;
}

/* The instances list of a query set (new ref) or NULL + error. */
static PyObject* query_instances(PyObject* query_set)
{
    return nd_attr(query_set, "instances");
}

/* int64 payload of instance.entity_id.value, or -1 with exc set. */
static int64_t instance_id_of(PyObject* instance)
{
    PyObject* id_value = nd_entity_id_value(instance);

    if (id_value == NULL) {
        return -1;
    }

    long long id = PyLong_AsLongLong(id_value);

    Py_DECREF(id_value);

    if (id == -1 && PyErr_Occurred()) {
        return -1;
    }

    return (int64_t) id;
}

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

ND_API NdStatusCode nd_query_new(NdDb* db, const char* struct_name,
                                 NdQuery** out_qs)
{
    if (db == NULL || struct_name == NULL || out_qs == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    *out_qs = NULL;

    ND_SHIM_BEGIN();

    PyObject* kwargs = Py_BuildValue("{s:s}", "struct_name", struct_name);
    PyObject* callable = kwargs != NULL ? nd_attr(db->facade, "query")
                                        : NULL;
    PyObject* query_set = callable != NULL
        ? nd_call(callable, NULL, kwargs) : NULL;

    Py_XDECREF(callable);
    Py_XDECREF(kwargs);

    if (query_set == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    NdQuery* handle = malloc(sizeof(NdQuery));

    if (handle == NULL) {
        Py_DECREF(query_set);
        nd_set_error("out of memory allocating query handle");
        ND_SHIM_RETURN(ND_ERR_MEMORY);
    }

    handle->query_set = query_set;
    *out_qs = handle;

    ND_SHIM_RETURN(ND_OK);
}

ND_API void nd_query_free(NdQuery* qs)
{
    if (qs == NULL) {
        return;
    }

    ND_SHIM_BEGIN();

    Py_CLEAR(qs->query_set);
    free(qs);

    ND_SHIM_RETURN_VOID();
}

/* ------------------------------------------------------------------ */
/* Filters                                                             */
/* ------------------------------------------------------------------ */

ND_API NdStatusCode nd_query_where_str(NdQuery* qs, const char* var,
                                       const char* expr)
{
    if (qs == NULL || var == NULL || expr == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    /* ExpressionParser.parse_text(expr) — the shared parser. */

    PyObject* parser = nd_engine_class(
        "nadata.parser.expressions.expression_parser", "ExpressionParser"
    );

    if (parser == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* text = Py_BuildValue("(s)", expr);
    PyObject* parse = text != NULL ? nd_attr(parser, "parse_text") : NULL;
    PyObject* expression = parse != NULL
        ? nd_call(parse, text, NULL) : NULL;

    Py_DECREF(parser);
    Py_XDECREF(parse);
    Py_XDECREF(text);

    if (expression == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    /* ExecutionScope() — fresh, empty scope. */

    PyObject* scope_cls = nd_engine_class(
        "nadata.core.expressions.scope", "ExecutionScope"
    );

    if (scope_cls == NULL) {
        Py_DECREF(expression);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* scope = nd_call(scope_cls, NULL, NULL);

    Py_DECREF(scope_cls);

    if (scope == NULL) {
        Py_DECREF(expression);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* kwargs = Py_BuildValue("{s:s,s:O,s:O}",
                                     "candidate_var", var,
                                     "expr", expression,
                                     "scope", scope);

    Py_DECREF(expression);
    Py_DECREF(scope);

    if (kwargs == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* filtered = call_query(qs->query_set, "where_expr", NULL,
                                    kwargs);

    Py_DECREF(kwargs);

    if (filtered == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    rebind(qs, filtered);

    ND_SHIM_RETURN(ND_OK);
}

ND_API NdStatusCode nd_query_where_cb(NdQuery* qs, NdPredicateFn fn,
                                      void* user_data)
{
    if (qs == NULL || fn == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* instances = query_instances(qs->query_set);

    if (instances == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* kept = PyList_New(0);

    if (kept == NULL) {
        Py_DECREF(instances);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    Py_ssize_t count = PyList_Size(instances);

    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject* instance = PyList_GetItem(instances, i);  /* borrowed */

        if (instance == NULL) {
            Py_DECREF(kept);
            Py_DECREF(instances);
            ND_SHIM_RETURN(nd_take_pyerror());
        }

        int64_t id = instance_id_of(instance);

        if (id < 0 && PyErr_Occurred()) {
            Py_DECREF(kept);
            Py_DECREF(instances);
            ND_SHIM_RETURN(nd_take_pyerror());
        }

        int32_t verdict = fn(id, user_data);

        if (verdict < 0) {
            Py_DECREF(kept);
            Py_DECREF(instances);
            nd_set_error("predicate callback aborted with status %d",
                         (int) verdict);
            ND_SHIM_RETURN((NdStatusCode) verdict);
        }

        if (verdict == 1) {
            if (PyList_Append(kept, instance) != 0) {
                Py_DECREF(kept);
                Py_DECREF(instances);
                ND_SHIM_RETURN(nd_take_pyerror());
            }
        }
    }

    Py_DECREF(instances);

    /* Rebuild: QuerySet(struct=..., instances=kept, database=..., query_executor=...) */
    PyObject* queryset_cls = nd_engine_class(
        "nadata.core.database.queryset", "QuerySet"
    );

    if (queryset_cls == NULL) {
        Py_DECREF(kept);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* struct_obj = nd_attr(qs->query_set, "struct");
    PyObject* database = nd_attr(qs->query_set, "database");
    PyObject* executor = nd_attr(qs->query_set, "query_executor");

    PyObject* kwargs = PyDict_New();

    if (kwargs == NULL ||
        PyDict_SetItemString(kwargs, "struct", struct_obj) != 0 ||
        PyDict_SetItemString(kwargs, "instances", kept) != 0 ||
        PyDict_SetItemString(kwargs, "database", database) != 0 ||
        PyDict_SetItemString(kwargs, "query_executor", executor) != 0) {
        Py_DECREF(kwargs);
        Py_XDECREF(struct_obj);
        Py_XDECREF(database);
        Py_XDECREF(executor);
        Py_DECREF(kept);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    /* The dict holds its own references now. */
    Py_DECREF(struct_obj);
    Py_DECREF(database);
    Py_DECREF(executor);

    PyObject* rebuilt = nd_call(queryset_cls, NULL, kwargs);

    Py_DECREF(kwargs);

    if (rebuilt == NULL) {
        Py_DECREF(kept);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    Py_DECREF(kept);  /* the rebuilt QuerySet owns its instance list */
    rebind(qs, rebuilt);

    ND_SHIM_RETURN(ND_OK);
}

ND_API NdStatusCode nd_query_order_by(NdQuery* qs, const char* attribute,
                                      int32_t descending)
{
    if (qs == NULL || attribute == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* kwargs = Py_BuildValue("{s:s,s:O}",
                                     "attribute", attribute,
                                     "descending", descending
                                         ? Py_True : Py_False);
    PyObject* ordered = kwargs != NULL
        ? call_query(qs->query_set, "order_by", NULL, kwargs) : NULL;

    Py_XDECREF(kwargs);

    if (ordered == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    rebind(qs, ordered);

    ND_SHIM_RETURN(ND_OK);
}

ND_API NdStatusCode nd_query_limit(NdQuery* qs, int64_t n)
{
    if (qs == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    if (n <= 0) {
        nd_set_error("limit must be >= 1, got %lld", (long long) n);
        ND_SHIM_RETURN(ND_ERR_INVALID_ARG);
    }

    PyObject* args = Py_BuildValue("(L)", (long long) n);
    PyObject* limited = args != NULL
        ? call_query(qs->query_set, "limit", args, NULL) : NULL;

    Py_XDECREF(args);

    if (limited == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    rebind(qs, limited);

    ND_SHIM_RETURN(ND_OK);
}

/* ------------------------------------------------------------------ */
/* Readers                                                             */
/* ------------------------------------------------------------------ */

/* Calls a no-argument QuerySet method returning a Python value. */
static PyObject* read_query_value(NdQuery* qs, const char* method,
                                  const char* attribute)
{
    PyObject* kwargs = attribute != NULL
        ? Py_BuildValue("{s:s}", "attribute", attribute) : NULL;
    PyObject* result = call_query(qs->query_set, method, NULL, kwargs);

    Py_XDECREF(kwargs);

    return result;  /* NULL with exc set on failure */
}

ND_API NdStatusCode nd_query_count(NdQuery* qs, int64_t* out)
{
    if (qs == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* result = read_query_value(qs, "count", NULL);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    long long count = PyLong_AsLongLong(result);

    Py_DECREF(result);

    if (count == -1 && PyErr_Occurred()) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    *out = (int64_t) count;

    ND_SHIM_RETURN(ND_OK);
}

ND_API NdStatusCode nd_query_exists(NdQuery* qs, int32_t* out)
{
    if (qs == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* result = read_query_value(qs, "exists", NULL);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    int truthy = PyObject_IsTrue(result);

    Py_DECREF(result);

    if (truthy < 0) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    *out = (int32_t) truthy;

    ND_SHIM_RETURN(ND_OK);
}

/* Shared first/last body; empty set -> ND_ERR_NOTFOUND (contract). */
static NdStatusCode first_or_last_id(NdQuery* qs, const char* method,
                                     int64_t* out_id)
{
    PyObject* result = call_query(qs->query_set, method, NULL, NULL);

    if (result == NULL) {
        return nd_take_pyerror();
    }

    if (result == Py_None) {
        Py_DECREF(result);
        nd_set_error("query set is empty; no %s instance", method);

        return ND_ERR_NOTFOUND;
    }

    int64_t id = instance_id_of(result);

    Py_DECREF(result);

    if (id < 0 && PyErr_Occurred()) {
        return nd_take_pyerror();
    }

    *out_id = id;

    return ND_OK;
}

ND_API NdStatusCode nd_query_first_id(NdQuery* qs, int64_t* out_id)
{
    if (qs == NULL || out_id == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    ND_SHIM_RETURN(first_or_last_id(qs, "first", out_id));
}

ND_API NdStatusCode nd_query_last_id(NdQuery* qs, int64_t* out_id)
{
    if (qs == NULL || out_id == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    ND_SHIM_RETURN(first_or_last_id(qs, "last", out_id));
}

ND_API NdStatusCode nd_query_pluck(NdQuery* qs, const char* attribute,
                                   NdValue* out)
{
    if (qs == NULL || attribute == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* result = read_query_value(qs, "pluck", attribute);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    NdStatusCode status = nd_value_from_py(result, out);

    Py_DECREF(result);

    ND_SHIM_RETURN(status);
}

ND_API NdStatusCode nd_query_sum_of(NdQuery* qs, const char* attribute,
                                    NdValue* out)
{
    if (qs == NULL || attribute == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* result = read_query_value(qs, "sum_of", attribute);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    NdStatusCode status = nd_value_from_py(result, out);

    Py_DECREF(result);

    ND_SHIM_RETURN(status);
}

ND_API NdStatusCode nd_query_min_of(NdQuery* qs, const char* attribute,
                                    NdValue* out)
{
    if (qs == NULL || attribute == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* result = read_query_value(qs, "min_of", attribute);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    NdStatusCode status = nd_value_from_py(result, out);

    Py_DECREF(result);

    ND_SHIM_RETURN(status);
}

ND_API NdStatusCode nd_query_max_of(NdQuery* qs, const char* attribute,
                                    NdValue* out)
{
    if (qs == NULL || attribute == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* result = read_query_value(qs, "max_of", attribute);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    NdStatusCode status = nd_value_from_py(result, out);

    Py_DECREF(result);

    ND_SHIM_RETURN(status);
}

ND_API NdStatusCode nd_query_average_of(NdQuery* qs, const char* attribute,
                                        double* out)
{
    if (qs == NULL || attribute == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* result = read_query_value(qs, "average_of", attribute);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    double average = PyFloat_AsDouble(result);

    Py_DECREF(result);

    if (average == -1.0 && PyErr_Occurred()) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    *out = average;

    ND_SHIM_RETURN(ND_OK);
}

ND_API NdStatusCode nd_query_to_list_ids(NdQuery* qs, NdValue* out)
{
    if (qs == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* instances = read_query_value(qs, "to_list", NULL);

    if (instances == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    /*
     * to_list returns StructInstance objects; convert each to its
     * entity id so the caller receives ND_ARRAY of ND_INSTANCE.
     */
    Py_ssize_t count = PyList_Size(instances);

    if (count < 0) {
        Py_DECREF(instances);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* ids = PyList_New(0);

    if (ids == NULL) {
        Py_DECREF(instances);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject* instance = PyList_GetItem(instances, i);  /* borrowed */
        PyObject* id_value = instance != NULL
            ? nd_entity_id_value(instance) : NULL;

        if (id_value == NULL || PyList_Append(ids, id_value) != 0) {
            Py_XDECREF(id_value);
            Py_DECREF(ids);
            Py_DECREF(instances);
            ND_SHIM_RETURN(nd_take_pyerror());
        }

        Py_DECREF(id_value);
    }

    Py_DECREF(instances);

    /* Wrap the raw ints into EntityId objects so nd_value_from_py
     * emits ND_INSTANCE items (the documented payload type). */
    PyObject* entity_id_cls = nd_engine_class(
        "nadata.core.database.entity_id", "EntityId"
    );

    if (entity_id_cls == NULL) {
        Py_DECREF(ids);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    Py_ssize_t id_count = PyList_Size(ids);

    for (Py_ssize_t i = 0; i < id_count; i++) {
        PyObject* raw_id = PyList_GetItem(ids, i);          /* borrowed */
        PyObject* call_args = Py_BuildValue("(O)", raw_id);
        PyObject* entity_id = call_args != NULL
            ? nd_call(entity_id_cls, call_args, NULL) : NULL;

        Py_XDECREF(call_args);

        if (entity_id == NULL || PyList_SetItem(ids, i, entity_id) != 0) {
            /* SetItem only fails on a non-list; entity_id is consumed
             * by SetItem on success, else leaked here. */
            Py_XDECREF(entity_id);
            Py_DECREF(entity_id_cls);
            Py_DECREF(ids);
            ND_SHIM_RETURN(nd_take_pyerror());
        }
    }

    Py_DECREF(entity_id_cls);

    NdStatusCode status = nd_value_from_py(ids, out);

    Py_DECREF(ids);

    ND_SHIM_RETURN(status);
}
