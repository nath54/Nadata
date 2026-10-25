/*
 * shim_choice_b.c — the by-entity-id API layer (ABI 0.2 additions).
 *
 * The facade-mapping surface works through SYMBOL names; this layer
 * works through the 64-bit ENTITY IDS returned by nd_get_symbol_id,
 * nd_query_first_id, nd_query_to_list_ids, ... — skipping the symbol
 * lookup and enabling a coherent "handle = object reference" model
 * in host bindings.
 *
 * All functions are append-only additions: no existing symbol changed
 * (ND_ABI_VERSION bumped 1 -> 2 for feature detection).
 *
 * Engine mapping (all against the LIVE database):
 *   nd_get_attr_by_id      instance.get_attribute + api._unwrap_value
 *   nd_set_attr_by_id      api._wrap_argument + instance.set_attribute
 *   nd_call_method_by_id   instance.call_method with the facade's
 *                          engine globals (defaults need them)
 *   nd_list_attr           instance.struct.get_all_attributes().keys()
 *   nd_list_methods        instance.struct.get_all_methods().keys()
 *   nd_instance_id_valid   database.get_by_id(...) is not None
 *   nd_get_by_nid          database.get_by_nid(struct_name, nid)
 */

/* Import Modules */
#include "shim_internal.h"



/* ------------------------------------------------------------------ */
/* nadata.api helpers (wrap/unwrap engine <-> plain values)            */
/* ------------------------------------------------------------------ */

/* Borrowed reference to a cached nadata.api helper, or NULL + exc. */
static PyObject* api_function(const char* name)
{
    static PyObject* api_module = NULL;   /* interpreter-lifetime */

    if (api_module == NULL) {
        api_module = PyImport_ImportModule("nadata.api");

        if (api_module == NULL) {
            return NULL;
        }
    }

    return PyObject_GetAttrString(api_module, name);
}

/* Calls fn(*one_arg) — new ref, or NULL with exc set. */
static PyObject* call_unary(PyObject* fn, PyObject* one_arg)
{
    PyObject* args = PyTuple_Pack(1, one_arg);

    if (args == NULL) {
        return NULL;
    }

    PyObject* result = nd_call(fn, args, NULL);

    Py_DECREF(args);

    return result;
}

/* The facade's engine globals (creates them lazily on first use). */
static PyObject* engine_globals(NdDb* db)
{
    PyObject* ensure = nd_attr(db->facade, "_ensure_globals");

    if (ensure != NULL) {
        PyObject* none = nd_call(ensure, NULL, NULL);

        Py_DECREF(ensure);

        if (none == NULL) {
            return NULL;
        }

        Py_DECREF(none);
    }

    return nd_attr(db->facade, "_globals");
}

/* The live database (new ref), or NULL with exc set. */
static PyObject* live_database(NdDb* db)
{
    return nd_facade_database(db->facade);
}

/* Resolves an entity id to its instance, or NULL + KeyError set. */
static PyObject* instance_by_id(NdDb* db, int64_t id)
{
    PyObject* database = live_database(db);

    if (database == NULL) {
        return NULL;
    }

    PyObject* instance = nd_instance_from_id(database, id);

    Py_DECREF(database);

    return instance;
}

/* ------------------------------------------------------------------ */
/* Attribute access by id                                              */
/* ------------------------------------------------------------------ */

ND_API NdStatusCode nd_get_attr_by_id(NdDb* db, int64_t id,
                                      const char* name, NdValue* out)
{
    if (db == NULL || name == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* instance = instance_by_id(db, id);

    if (instance == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* getter = nd_attr(instance, "get_attribute");

    if (getter == NULL) {
        Py_DECREF(instance);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* args = Py_BuildValue("(s)", name);
    PyObject* engine_value = args != NULL
        ? nd_call(getter, args, NULL) : NULL;

    Py_DECREF(args);
    Py_DECREF(getter);

    if (engine_value == NULL) {
        Py_DECREF(instance);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* unwrap = api_function("_unwrap_value");
    PyObject* plain = unwrap != NULL
        ? call_unary(unwrap, engine_value) : NULL;

    Py_XDECREF(unwrap);
    Py_DECREF(engine_value);

    if (plain == NULL) {
        Py_DECREF(instance);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    NdStatusCode status = nd_value_from_py(plain, out);

    Py_DECREF(plain);
    Py_DECREF(instance);

    if (status != ND_OK) {
        ND_SHIM_RETURN(status);
    }

    ND_SHIM_RETURN(ND_OK);
}

ND_API NdStatusCode nd_set_attr_by_id(NdDb* db, int64_t id,
                                      const char* name,
                                      const NdValue* value)
{
    if (db == NULL || name == NULL || value == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* instance = instance_by_id(db, id);

    if (instance == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* database = live_database(db);

    if (database == NULL) {
        Py_DECREF(instance);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* wrapped = NULL;
    NdStatusCode status = nd_py_from_value(database, value, &wrapped);

    Py_DECREF(database);

    if (status != ND_OK) {
        Py_DECREF(instance);
        ND_SHIM_RETURN(status);
    }

    PyObject* wrap = api_function("_wrap_argument");
    PyObject* engine_value = wrap != NULL
        ? call_unary(wrap, wrapped) : NULL;

    Py_XDECREF(wrap);
    Py_DECREF(wrapped);

    if (engine_value == NULL) {
        Py_DECREF(instance);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* setter = nd_attr(instance, "set_attribute");

    if (setter == NULL) {
        Py_DECREF(engine_value);
        Py_DECREF(instance);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* kwargs = Py_BuildValue("{s:s,s:O}",
                                     "name", name, "value", engine_value);

    Py_DECREF(engine_value);

    if (kwargs == NULL) {
        Py_DECREF(setter);
        Py_DECREF(instance);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* none = nd_call(setter, NULL, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(setter);
    Py_DECREF(instance);

    if (none == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    Py_DECREF(none);

    ND_SHIM_RETURN(ND_OK);
}

/* ------------------------------------------------------------------ */
/* Method dispatch by id                                               */
/* ------------------------------------------------------------------ */

ND_API NdStatusCode nd_call_method_by_id(NdDb* db, int64_t id,
                                         const char* name,
                                         const NdValue* args,
                                         const NdValue* kwargs,
                                         NdValue* out)
{
    if (db == NULL || name == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* instance = instance_by_id(db, id);

    if (instance == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* database = live_database(db);

    if (database == NULL) {
        Py_DECREF(instance);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    /* Convert the borrowed inputs to plain Python values. */
    PyObject* py_args = NULL;
    PyObject* py_kwargs = NULL;
    NdStatusCode status = nd_py_from_args(database, args, kwargs,
                                          &py_args, &py_kwargs);

    Py_DECREF(database);

    if (status != ND_OK) {
        Py_DECREF(instance);
        ND_SHIM_RETURN(status);
    }


    PyObject* wrap = api_function("_wrap_argument");

    if (wrap == NULL) {
        Py_XDECREF(py_args);
        Py_XDECREF(py_kwargs);
        Py_DECREF(instance);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    /* Wrap every element (the engine expects wrapped Values). */
    PyObject* call_args = PyList_New(0);

    if (call_args == NULL) {
        Py_DECREF(wrap);
        Py_XDECREF(py_args);
        Py_XDECREF(py_kwargs);
        Py_DECREF(instance);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    /* py_args may be a tuple (no positional args) or a list — use the
     * sequence protocol so both iterate without raising SystemError. */
    Py_ssize_t arg_count = py_args != NULL
        ? PySequence_Size(py_args) : 0;

    if (arg_count < 0) {
        PyErr_Clear();
        arg_count = 0;
    }

    for (Py_ssize_t i = 0; i < arg_count; i++) {
        PyObject* item = PySequence_GetItem(py_args, i);    /* new ref */
        PyObject* wrapped = call_unary(wrap, item);

        if (wrapped == NULL || PyList_Append(call_args, wrapped) != 0) {
            Py_XDECREF(wrapped);
            Py_XDECREF(item);
            Py_DECREF(call_args);
            Py_DECREF(wrap);
            Py_XDECREF(py_args);
            Py_XDECREF(py_kwargs);
            Py_DECREF(instance);
            ND_SHIM_RETURN(nd_take_pyerror());
        }

        Py_DECREF(wrapped);
        Py_DECREF(item);
    }

    PyObject* call_kwargs = PyDict_New();

    if (call_kwargs == NULL) {
        Py_DECREF(call_args);
        Py_DECREF(wrap);
        Py_XDECREF(py_args);
        Py_XDECREF(py_kwargs);
        Py_DECREF(instance);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    if (py_kwargs != NULL) {
        Py_ssize_t pos = 0;
        PyObject* key = NULL;
        PyObject* item = NULL;

        while (PyDict_Next(py_kwargs, &pos, &key, &item)) {
            PyObject* wrapped = call_unary(wrap, item);

            if (wrapped == NULL
                || PyDict_SetItem(call_kwargs, key, wrapped) != 0) {
                Py_XDECREF(wrapped);
                Py_DECREF(call_kwargs);
                Py_DECREF(call_args);
                Py_DECREF(wrap);
                Py_XDECREF(py_args);
                Py_XDECREF(py_kwargs);
                Py_DECREF(instance);
                ND_SHIM_RETURN(nd_take_pyerror());
            }

            Py_DECREF(wrapped);
        }
    }

    Py_DECREF(wrap);
    Py_XDECREF(py_args);
    Py_XDECREF(py_kwargs);


    PyObject* globals = engine_globals(db);

    if (globals == NULL) {
        Py_DECREF(call_kwargs);
        Py_DECREF(call_args);
        Py_DECREF(instance);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* method_kwargs = Py_BuildValue(
        "{s:s,s:O,s:O,s:O}",
        "name", name, "args", call_args, "kwargs", call_kwargs,
        "globals_dict", globals);

    Py_DECREF(call_args);
    Py_DECREF(call_kwargs);
    Py_DECREF(globals);

    if (method_kwargs == NULL) {
        Py_DECREF(instance);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* method = nd_attr(instance, "call_method");
    PyObject* result = method != NULL
        ? nd_call(method, NULL, method_kwargs) : NULL;

    Py_XDECREF(method);
    Py_DECREF(method_kwargs);

    if (result == NULL) {
        Py_DECREF(instance);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    /* instance.call_method returns the engine's Value wrapper; the
     * facade unwraps it (api._unwrap_value) before exposing plain
     * values — do the same so results cross the ABI boundary. */
    PyObject* unwrap = api_function("_unwrap_value");
    PyObject* plain = unwrap != NULL
        ? call_unary(unwrap, result) : NULL;

    Py_XDECREF(unwrap);
    Py_DECREF(result);

    if (plain == NULL) {
        Py_DECREF(instance);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    status = nd_value_from_py(plain, out);

    Py_DECREF(plain);
    Py_DECREF(instance);

    if (status != ND_OK) {
        ND_SHIM_RETURN(status);
    }

    ND_SHIM_RETURN(ND_OK);
}

/* ------------------------------------------------------------------ */
/* Introspection by id                                                 */
/* ------------------------------------------------------------------ */

/* Shared body for nd_list_attr / nd_list_methods. */
static NdStatusCode list_members(NdDb* db, int64_t id,
                                 const char* getter_name, NdValue* out)
{
    PyObject* instance = instance_by_id(db, id);

    if (instance == NULL) {
        return nd_take_pyerror();
    }

    PyObject* struct_obj = nd_attr(instance, "struct");

    if (struct_obj == NULL) {
        Py_DECREF(instance);
        return nd_take_pyerror();
    }

    PyObject* getter = nd_attr(struct_obj, getter_name);
    PyObject* members = getter != NULL
        ? nd_call(getter, NULL, NULL) : NULL;

    Py_DECREF(getter);

    if (members == NULL) {
        Py_DECREF(struct_obj);
        Py_DECREF(instance);
        return nd_take_pyerror();
    }

    PyObject* names = PyDict_Keys(members);

    Py_DECREF(members);

    if (names == NULL) {
        Py_DECREF(struct_obj);
        Py_DECREF(instance);
        return nd_take_pyerror();
    }

    NdStatusCode status = nd_value_from_py(names, out);

    Py_DECREF(names);
    Py_DECREF(struct_obj);
    Py_DECREF(instance);

    return status;
}

ND_API NdStatusCode nd_list_attr(NdDb* db, int64_t id, NdValue* out)
{
    if (db == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    ND_SHIM_RETURN(list_members(db, id, "get_all_attributes", out));
}

ND_API NdStatusCode nd_list_methods(NdDb* db, int64_t id, NdValue* out)
{
    if (db == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    ND_SHIM_RETURN(list_members(db, id, "get_all_methods", out));
}

ND_API NdStatusCode nd_instance_id_valid(NdDb* db, int64_t id,
                                         int32_t* out)
{
    if (db == NULL || out == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* database = live_database(db);

    if (database == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* getter = nd_attr(database, "get_by_id");

    if (getter == NULL) {
        Py_DECREF(database);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* call_args = Py_BuildValue("(K)", (unsigned long long) id);
    PyObject* instance = call_args != NULL
        ? nd_call(getter, call_args, NULL) : NULL;

    Py_XDECREF(call_args);
    Py_DECREF(getter);
    Py_DECREF(database);

    if (instance == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    *out = (instance != Py_None) ? 1 : 0;

    Py_DECREF(instance);

    ND_SHIM_RETURN(ND_OK);
}

/* ------------------------------------------------------------------ */
/* Named-id lookup                                                     */
/* ------------------------------------------------------------------ */

ND_API NdStatusCode nd_get_by_nid(NdDb* db, const char* base_struct,
                                  const char* nid, int64_t* out_id)
{
    if (db == NULL || base_struct == NULL || nid == NULL || out_id == NULL) {
        return ND_ERR_INVALID_ARG;
    }

    ND_SHIM_BEGIN();

    PyObject* database = live_database(db);

    if (database == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* getter = nd_attr(database, "get_by_nid");

    if (getter == NULL) {
        Py_DECREF(database);
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* kwargs = Py_BuildValue("{s:s,s:s}",
                                     "struct_name", base_struct,
                                     "nid", nid);

    Py_DECREF(database);

    if (kwargs == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* instance = nd_call(getter, NULL, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(getter);

    if (instance == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    if (instance == Py_None) {
        Py_DECREF(instance);
        PyErr_Format(PyExc_KeyError, "no instance for nid '%s/%s'",
                     base_struct, nid);

        ND_SHIM_RETURN(nd_take_pyerror());
    }

    PyObject* id_value = nd_entity_id_value(instance);

    Py_DECREF(instance);

    if (id_value == NULL) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    long long value = PyLong_AsLongLong(id_value);

    Py_DECREF(id_value);

    if (value == -1 && PyErr_Occurred()) {
        ND_SHIM_RETURN(nd_take_pyerror());
    }

    *out_id = (int64_t) value;

    ND_SHIM_RETURN(ND_OK);
}
