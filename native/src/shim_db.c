/*
 * shim_db.c — facade-facing ABI half: lifecycle, introspection,
 * instances, persistence and code execution.
 *
 * Every function maps 1:1 onto a nadata.api.Nadata method (Choice A).
 * Queries live in shim_query.c.
 */

/* Import Modules */
#include "shim_internal.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Internal helpers                                                    */
/* ------------------------------------------------------------------ */

static NdStatusCode new_db(PyObject* facade, NdDb** out_db)
{
    NdDb* db = malloc(sizeof(NdDb));

    if (db == NULL) {
        Py_DECREF(facade);
        nd_set_error("out of memory allocating database handle");
        return ND_ERR_MEMORY;
    }

    db->facade = facade;  /* handle owns one reference */
    *out_db = db;

    return ND_OK;
}

/* Calls a facade method by name with a keyword dict; new ref or NULL. */
static PyObject* call_facade(NdDb* db, const char* method,
                             PyObject* kwargs)
{
    PyObject* callable = nd_attr(db->facade, method);

    if (callable == NULL) {
        return NULL;
    }

    PyObject* result = nd_call(callable, NULL, kwargs);

    Py_DECREF(callable);

    return result;
}

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

ND_API NdStatusCode nd_load(const char* path, NdDb** out_db)
{
    if (path == NULL || out_db == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    *out_db = NULL;

    ND_SHIM_BEGIN();

    if (nd_ensure_interpreter() != 0) {
        ND_SHIM_RETURN(ND_ERR_RUNTIME);
    }

    /* Nadata.load(path) — via the classmethod, NOT the constructor. */
    PyObject* load = nd_attr(nd_nadata_cls(), "load");

    if (load == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* args = Py_BuildValue("(s)", path);
    PyObject* facade = args != NULL ? nd_call(load, args, NULL) : NULL;

    Py_DECREF(load);
    Py_XDECREF(args);

    if (facade == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    NdStatusCode status = new_db(facade, out_db);

    ND_SHIM_RETURN(status);
}

ND_API NdStatusCode nd_open(const char* storage_path, NdDb** out_db)
{
    if (storage_path == NULL || out_db == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    *out_db = NULL;

    ND_SHIM_BEGIN();

    if (nd_ensure_interpreter() != 0) {
        ND_SHIM_RETURN(ND_ERR_RUNTIME);
    }

    PyObject* kwargs = Py_BuildValue("{s:s}", "storage_path", storage_path);

    if (kwargs == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* callable = nd_attr(nd_nadata_cls(), "open");
    PyObject* facade = callable != NULL
        ? nd_call(callable, NULL, kwargs) : NULL;

    Py_XDECREF(callable);
    Py_DECREF(kwargs);

    if (facade == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    NdStatusCode status = new_db(facade, out_db);

    ND_SHIM_RETURN(status);
}

ND_API NdStatusCode nd_open_folder(const char* folder_path,
                                   const char* db_path,
                                   int32_t run_auto, int32_t force_init,
                                   NdDb** out_db)
{
    if (folder_path == NULL || out_db == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    *out_db = NULL;

    ND_SHIM_BEGIN();

    if (nd_ensure_interpreter() != 0) {
        ND_SHIM_RETURN(ND_ERR_RUNTIME);
    }

    PyObject* kwargs = PyDict_New();

    if (kwargs == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* folder = PyUnicode_FromString(folder_path);
    PyObject* db = db_path != NULL
        ? PyUnicode_FromString(db_path) : Py_NewRef(Py_None);
    PyObject* auto_flag = PyBool_FromLong(run_auto);
    PyObject* init_flag = PyBool_FromLong(force_init);

    if (folder == NULL || db == NULL || auto_flag == NULL ||
        init_flag == NULL ||
        PyDict_SetItemString(kwargs, "folder_path", folder) != 0 ||
        PyDict_SetItemString(kwargs, "db_path", db) != 0 ||
        PyDict_SetItemString(kwargs, "run_auto", auto_flag) != 0 ||
        PyDict_SetItemString(kwargs, "force_init", init_flag) != 0) {
        Py_XDECREF(folder);
        Py_XDECREF(db);
        Py_XDECREF(auto_flag);
        Py_XDECREF(init_flag);
        Py_DECREF(kwargs);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    Py_DECREF(folder);
    Py_DECREF(db);
    Py_DECREF(auto_flag);
    Py_DECREF(init_flag);

    PyObject* callable = nd_attr(nd_nadata_cls(), "open_folder");
    PyObject* facade = callable != NULL
        ? nd_call(callable, NULL, kwargs) : NULL;

    Py_XDECREF(callable);
    Py_DECREF(kwargs);

    if (facade == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    NdStatusCode status = new_db(facade, out_db);

    ND_SHIM_RETURN(status);
}

ND_API void nd_close(NdDb* db)
{
    if (db == NULL) {
        return;
    }

    ND_SHIM_BEGIN();

    Py_CLEAR(db->facade);
    free(db);

    ND_SHIM_RETURN_VOID();
}

/* ------------------------------------------------------------------ */
/* Introspection                                                       */
/* ------------------------------------------------------------------ */

/*
 * Reads a list-of-strings facade property into an ND_ARRAY NdValue.
 * Shared by the three *_names introspection calls.
 */
static NdStatusCode names_to_value(NdDb* db, const char* property,
                                   NdValue* out)
{
    PyObject* names = nd_attr(db->facade, property);

    if (names == NULL) {
        return nd_take_pyerror();
    }

    NdStatusCode status = nd_value_from_py(names, out);

    Py_DECREF(names);

    return status;
}

ND_API NdStatusCode nd_struct_names(NdDb* db, NdValue* out)
{
    if (db == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    ND_SHIM_RETURN(names_to_value(db, "struct_names", out));
}

ND_API NdStatusCode nd_function_names(NdDb* db, NdValue* out)
{
    if (db == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    ND_SHIM_RETURN(names_to_value(db, "function_names", out));
}

ND_API NdStatusCode nd_symbol_names(NdDb* db, NdValue* out)
{
    if (db == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    ND_SHIM_RETURN(names_to_value(db, "symbol_names", out));
}

ND_API NdStatusCode nd_summary(NdDb* db, char** out)
{
    if (db == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    *out = NULL;

    ND_SHIM_BEGIN();

    PyObject* result = call_facade(db, "summary", NULL);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    const char* text = PyUnicode_AsUTF8(result);

    if (text == NULL) {
        Py_DECREF(result);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    char* copy = nd_strdup(text);

    Py_DECREF(result);

    if (copy == NULL) {
        nd_set_error("out of memory copying summary");
        ND_SHIM_RETURN(ND_ERR_MEMORY);
    }

    *out = copy;

    ND_SHIM_RETURN(ND_OK);
}

ND_API NdStatusCode nd_count(NdDb* db, const char* struct_name, int64_t* out)
{
    if (db == NULL || struct_name == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* kwargs = Py_BuildValue("{s:s}", "struct_name", struct_name);
    PyObject* result = kwargs != NULL ? call_facade(db, "count", kwargs)
                                      : NULL;

    Py_XDECREF(kwargs);

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

ND_API NdStatusCode nd_total_instance_count(NdDb* db, int64_t* out)
{
    if (db == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* result = call_facade(db, "total_instance_count", NULL);

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

/* ------------------------------------------------------------------ */
/* Instances and attributes                                            */
/* ------------------------------------------------------------------ */

ND_API NdStatusCode nd_get_symbol_id(NdDb* db, const char* symbol,
                                     int64_t* out_id)
{
    if (db == NULL || symbol == NULL || out_id == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* instance = call_facade(
        db, "get", Py_BuildValue("{s:s}", "symbol", symbol)
    );

    if (instance == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* id_value = nd_entity_id_value(instance);

    Py_DECREF(instance);

    if (id_value == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    long long id = PyLong_AsLongLong(id_value);

    Py_DECREF(id_value);

    if (id == -1 && PyErr_Occurred()) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    *out_id = (int64_t) id;

    ND_SHIM_RETURN(ND_OK);
}

ND_API NdStatusCode nd_get_attribute(NdDb* db, const char* symbol,
                                     const char* name, NdValue* out)
{
    if (db == NULL || symbol == NULL || name == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* kwargs = Py_BuildValue("{s:s,s:s}",
                                     "symbol", symbol, "name", name);
    PyObject* result = kwargs != NULL
        ? call_facade(db, "get_attribute", kwargs) : NULL;

    Py_XDECREF(kwargs);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    NdStatusCode status = nd_value_from_py(result, out);

    Py_DECREF(result);

    ND_SHIM_RETURN(status);
}

ND_API NdStatusCode nd_set_attribute(NdDb* db, const char* symbol,
                                     const char* name,
                                     const NdValue* value)
{
    if (db == NULL || symbol == NULL || name == NULL || value == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* database = nd_facade_database(db->facade);

    if (database == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* wrapped = NULL;
    NdStatusCode status = nd_py_from_value(database, value, &wrapped);

    Py_DECREF(database);

    if (status != ND_OK) {
        ND_SHIM_RETURN(status);
    }

    PyObject* kwargs = Py_BuildValue("{s:s,s:s,s:O}",
                                     "symbol", symbol, "name", name,
                                     "value", wrapped);

    if (kwargs == NULL) {
        Py_DECREF(wrapped);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    /* The dict holds its own reference now. */
    Py_DECREF(wrapped);

    PyObject* result = call_facade(db, "set_attribute", kwargs);

    Py_DECREF(kwargs);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    Py_DECREF(result);

    ND_SHIM_RETURN(ND_OK);
}

ND_API NdStatusCode nd_create_instance(NdDb* db, const char* struct_name,
                                       const NdValue* attributes,
                                       const char* symbol, int64_t* out_id)
{
    if (db == NULL || struct_name == NULL || out_id == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* kwargs = PyDict_New();
    PyObject* name_key = PyUnicode_FromString("struct_name");
    PyObject* name_value = PyUnicode_FromString(struct_name);

    if (kwargs == NULL || name_key == NULL || name_value == NULL ||
        PyDict_SetItem(kwargs, name_key, name_value) != 0) {
        Py_XDECREF(name_key);
        Py_XDECREF(name_value);
        Py_XDECREF(kwargs);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    Py_DECREF(name_key);
    Py_DECREF(name_value);

    /* symbol: omit the key entirely when not binding. */
    if (symbol != NULL && symbol[0] != '\0') {
        PyObject* symbol_value = PyUnicode_FromString(symbol);

        if (symbol_value == NULL ||
            PyDict_SetItemString(kwargs, "symbol", symbol_value) != 0) {
            Py_XDECREF(symbol_value);
            Py_DECREF(kwargs);
            ND_SHIM_RETURN(nd_take_pyerror());
        }

        Py_DECREF(symbol_value);
    }

    /* attributes: pass through as a plain dict when present. */
    if (attributes != NULL && attributes->tag != ND_NULL) {
        PyObject* database = nd_facade_database(db->facade);
        PyObject* wrapped = NULL;

        if (database == NULL) {
            Py_DECREF(kwargs);
            ND_SHIM_RETURN(nd_take_pyerror());
        }

        NdStatusCode status = nd_py_from_value(database, attributes,
                                               &wrapped);

        Py_DECREF(database);

        if (status != ND_OK) {
            Py_DECREF(kwargs);
            ND_SHIM_RETURN(status);
        }

        if (PyDict_SetItemString(kwargs, "attributes", wrapped) != 0) {
            Py_DECREF(wrapped);
            Py_DECREF(kwargs);
            ND_SHIM_RETURN(nd_take_pyerror());
        }

        Py_DECREF(wrapped);
    }

    PyObject* instance = call_facade(db, "create_instance", kwargs);

    Py_DECREF(kwargs);

    if (instance == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* id_value = nd_entity_id_value(instance);

    Py_DECREF(instance);

    if (id_value == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    long long id = PyLong_AsLongLong(id_value);

    Py_DECREF(id_value);

    if (id == -1 && PyErr_Occurred()) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    *out_id = (int64_t) id;

    ND_SHIM_RETURN(ND_OK);
}

ND_API NdStatusCode nd_call_function(NdDb* db, const char* name,
                                     const NdValue* args,
                                     const NdValue* kwargs, NdValue* out)
{
    if (db == NULL || name == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* py_args = NULL;
    PyObject* py_kwargs = NULL;
    PyObject* database = nd_facade_database(db->facade);

    if (database == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    NdStatusCode status = nd_py_from_args(database, args, kwargs,
                                          &py_args, &py_kwargs);

    Py_DECREF(database);

    if (status != ND_OK) {
        ND_SHIM_RETURN(status);
    }

    PyObject* name_value = PyUnicode_FromString(name);

    if (name_value == NULL) {
        Py_XDECREF(py_args);
        Py_XDECREF(py_kwargs);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* callable = nd_attr(db->facade, "call_function");

    if (callable == NULL) {
        Py_DECREF(name_value);
        Py_XDECREF(py_args);
        Py_XDECREF(py_kwargs);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    /* call_function(name, *args, **kwargs) — prepend the name. */
    PyObject* head = PyTuple_Pack(1, name_value);

    Py_DECREF(name_value);

    if (head == NULL) {
        Py_DECREF(callable);
        Py_XDECREF(py_args);
        Py_XDECREF(py_kwargs);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* call_args = PySequence_List(head);

    Py_DECREF(head);

    if (call_args == NULL ||
        PySequence_InPlaceConcat(call_args, py_args) == NULL) {
        Py_XDECREF(call_args);
        Py_DECREF(callable);
        Py_XDECREF(py_args);
        Py_XDECREF(py_kwargs);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* result = nd_call(callable, call_args, py_kwargs);

    Py_DECREF(call_args);
    Py_DECREF(callable);
    Py_XDECREF(py_args);
    Py_XDECREF(py_kwargs);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    status = nd_value_from_py(result, out);

    Py_DECREF(result);

    ND_SHIM_RETURN(status);
}

ND_API NdStatusCode nd_call_method(NdDb* db, const char* symbol,
                                   const char* name, const NdValue* args,
                                   const NdValue* kwargs, NdValue* out)
{
    if (db == NULL || symbol == NULL || name == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* py_args = NULL;
    PyObject* py_kwargs = NULL;
    PyObject* database = nd_facade_database(db->facade);

    if (database == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    NdStatusCode status = nd_py_from_args(database, args, kwargs,
                                          &py_args, &py_kwargs);

    Py_DECREF(database);

    if (status != ND_OK) {
        ND_SHIM_RETURN(status);
    }

    /* call_method(symbol, name, *args, **kwargs) — list form so the
     * positional args (a list) concat cleanly. */
    PyObject* call_args = PyList_New(0);
    PyObject* symbol_value = PyUnicode_FromString(symbol);
    PyObject* name_value = PyUnicode_FromString(name);

    if (call_args == NULL || symbol_value == NULL || name_value == NULL ||
        PyList_Append(call_args, symbol_value) != 0 ||
        PyList_Append(call_args, name_value) != 0) {
        Py_XDECREF(call_args);
        Py_XDECREF(symbol_value);
        Py_XDECREF(name_value);
        Py_XDECREF(py_args);
        Py_XDECREF(py_kwargs);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    Py_DECREF(symbol_value);
    Py_DECREF(name_value);

    PyObject* extended = PySequence_InPlaceConcat(call_args, py_args);

    if (extended == NULL) {
        Py_DECREF(call_args);
        Py_XDECREF(py_args);
        Py_XDECREF(py_kwargs);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    /* InPlaceConcat may return the same or a new object: keep exactly
     * one owning reference in `extended`. */
    if (extended != call_args) {
        Py_DECREF(call_args);
    }

    PyObject* callable = nd_attr(db->facade, "call_method");

    if (callable == NULL) {
        Py_DECREF(extended);
        Py_XDECREF(py_args);
        Py_XDECREF(py_kwargs);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* result = nd_call(callable, extended, py_kwargs);

    Py_DECREF(extended);
    Py_DECREF(callable);
    Py_XDECREF(py_args);
    Py_XDECREF(py_kwargs);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    status = nd_value_from_py(result, out);

    Py_DECREF(result);

    ND_SHIM_RETURN(status);
}

/* ------------------------------------------------------------------ */
/* Persistence                                                         */
/* ------------------------------------------------------------------ */

ND_API NdStatusCode nd_dump(NdDb* db, const char* storage_path)
{
    if (db == NULL || storage_path == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* kwargs = Py_BuildValue("{s:s}", "storage_path", storage_path);
    PyObject* result = kwargs != NULL ? call_facade(db, "dump", kwargs)
                                      : NULL;

    Py_XDECREF(kwargs);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    Py_DECREF(result);

    ND_SHIM_RETURN(ND_OK);
}

ND_API NdStatusCode nd_save_changes(NdDb* db, const char* storage_path,
                                    int64_t* out_docs)
{
    if (db == NULL || storage_path == NULL || out_docs == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* kwargs = Py_BuildValue("{s:s}", "storage_path", storage_path);
    PyObject* result = kwargs != NULL
        ? call_facade(db, "save_changes", kwargs) : NULL;

    Py_XDECREF(kwargs);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    long long docs = PyLong_AsLongLong(result);

    Py_DECREF(result);

    if (docs == -1 && PyErr_Occurred()) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    *out_docs = (int64_t) docs;

    ND_SHIM_RETURN(ND_OK);
}

ND_API NdStatusCode nd_compact(NdDb* db, const char* output_path,
                               int64_t* out_before, int64_t* out_after)
{
    if (db == NULL || out_before == NULL || out_after == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* kwargs = PyDict_New();

    if (kwargs == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    if (output_path != NULL) {
        PyObject* path_value = PyUnicode_FromString(output_path);

        if (path_value == NULL ||
            PyDict_SetItemString(kwargs, "output_path", path_value) != 0) {
            Py_XDECREF(path_value);
            Py_DECREF(kwargs);
            ND_SHIM_RETURN(nd_take_pyerror());
        }

        Py_DECREF(path_value);
    }

    PyObject* result = call_facade(db, "compact", kwargs);

    Py_DECREF(kwargs);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* before = PySequence_GetItem(result, 0);
    PyObject* after = PySequence_GetItem(result, 1);

    Py_DECREF(result);

    if (before == NULL || after == NULL) {
        Py_XDECREF(before);
        Py_XDECREF(after);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    long long count_before = PyLong_AsLongLong(before);
    long long count_after = PyLong_AsLongLong(after);

    Py_DECREF(before);
    Py_DECREF(after);

    if ((count_before == -1 || count_after == -1) && PyErr_Occurred()) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    *out_before = (int64_t) count_before;
    *out_after = (int64_t) count_after;

    ND_SHIM_RETURN(ND_OK);
}

ND_API NdStatusCode nd_set_loop_limit(NdDb* db, int64_t limit)
{
    if (db == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* kwargs = Py_BuildValue("{s:L}", "limit", (long long) limit);
    PyObject* result = kwargs != NULL
        ? call_facade(db, "set_loop_limit", kwargs) : NULL;

    Py_XDECREF(kwargs);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    Py_DECREF(result);

    ND_SHIM_RETURN(ND_OK);
}

/* ------------------------------------------------------------------ */
/* Code execution                                                      */
/* ------------------------------------------------------------------ */

/* Shared body for execute_schema / execute_data (same call shape). */
static NdStatusCode execute_text(NdDb* db, const char* method,
                                 const char* text)
{
    const char* key = strcmp(method, "execute_data") == 0
        ? "data_text" : "schema_text";

    PyObject* kwargs = Py_BuildValue("{s:s}", key, text);

    PyObject* result = kwargs != NULL ? call_facade(db, method, kwargs)
                                      : NULL;

    Py_XDECREF(kwargs);

    if (result == NULL) {
        return nd_take_pyerror();
    }

    Py_DECREF(result);

    return ND_OK;
}

ND_API NdStatusCode nd_execute_schema(NdDb* db, const char* schema_text)
{
    if (db == NULL || schema_text == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    ND_SHIM_RETURN(execute_text(db, "execute_schema", schema_text));
}

ND_API NdStatusCode nd_execute_data(NdDb* db, const char* data_text)
{
    if (db == NULL || data_text == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    ND_SHIM_RETURN(execute_text(db, "execute_data", data_text));
}

ND_API NdStatusCode nd_execute(NdDb* db, const char* text)
{
    if (db == NULL || text == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* kwargs = Py_BuildValue("{s:s}", "text", text);
    PyObject* result = kwargs != NULL ? call_facade(db, "execute", kwargs)
                                      : NULL;

    Py_XDECREF(kwargs);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    Py_DECREF(result);

    ND_SHIM_RETURN(ND_OK);
}

ND_API NdStatusCode nd_execute_script_file(NdDb* db, const char* file_path,
                                           const NdValue* params)
{
    if (db == NULL || file_path == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* kwargs = PyDict_New();

    if (kwargs == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* path_value = PyUnicode_FromString(file_path);

    if (path_value == NULL ||
        PyDict_SetItemString(kwargs, "file_path", path_value) != 0) {
        Py_XDECREF(path_value);
        Py_DECREF(kwargs);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    Py_DECREF(path_value);

    if (params != NULL && params->tag != ND_NULL) {
        PyObject* database = nd_facade_database(db->facade);
        PyObject* wrapped = NULL;

        if (database == NULL) {
            Py_DECREF(kwargs);
            ND_SHIM_RETURN(nd_take_pyerror());
        }

        NdStatusCode status = nd_py_from_value(database, params, &wrapped);

        Py_DECREF(database);

        if (status != ND_OK) {
            Py_DECREF(kwargs);
            ND_SHIM_RETURN(status);
        }

        if (PyDict_SetItemString(kwargs, "params", wrapped) != 0) {
            Py_DECREF(wrapped);
            Py_DECREF(kwargs);
            ND_SHIM_RETURN(nd_take_pyerror());
        }

        Py_DECREF(wrapped);
    }

    PyObject* result = call_facade(db, "execute_script_file", kwargs);

    Py_DECREF(kwargs);

    if (result == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    Py_DECREF(result);

    ND_SHIM_RETURN(ND_OK);
}

/* ------------------------------------------------------------------ */
/* Errors                                                              */
/* ------------------------------------------------------------------ */

ND_API const char* nd_last_error(NdDb* db)
{
    (void) db;  /* reserved: per-db error channels arrive with Choice B */

    return nd_tls_error();
}
