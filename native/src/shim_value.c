/*
 * shim_value.c — NdValue <-> PyObject conversion.
 *
 * Ownership (mirrors native/nadata.h):
 *   - INPUT (const NdValue*) values are borrowed: nd_py_from_value
 *     creates fresh Python objects and never retains the C memory.
 *   - OUTPUT (NdValue*) values are caller-owned: nd_value_from_py
 *     heap-allocates strings/arrays/dicts; nd_free_value releases.
 */

/* Import Modules */
#include "shim_internal.h"

#include <string.h>

/* Home-type payloads are not representable in ABI v1 (contract). */
#define ND_HOME_TYPES_MODULE_PREFIX "nadata.core.types.home_types"

/* ------------------------------------------------------------------ */
/* Memory helpers                                                      */
/* ------------------------------------------------------------------ */

char* nd_strdup(const char* text)
{
    size_t length = strlen(text) + 1;
    char* copy = malloc(length);

    if (copy != NULL) {
        memcpy(copy, text, length);
    }

    return copy;
}

/* ------------------------------------------------------------------ */
/* Small Python helpers                                                */
/* ------------------------------------------------------------------ */

PyObject* nd_attr(PyObject* obj, const char* name)
{
    return PyObject_GetAttrString(obj, name);
}

PyObject* nd_call(PyObject* callable, PyObject* args, PyObject* kwargs)
{
    PyObject* owned_args = NULL;
    PyObject* result = NULL;

    /* PyObject_Call REQUIRES a tuple; NULL means an empty one, and
     * callers may hand other sequences. */
    if (args == NULL) {
        owned_args = PyTuple_New(0);
    } else if (!PyTuple_Check(args)) {
        owned_args = PySequence_Tuple(args);
    }

    if (owned_args != NULL) {
        args = owned_args;
    }

    result = PyObject_Call(callable, args, kwargs);

    Py_XDECREF(owned_args);

    return result;
}

PyObject* nd_facade_database(PyObject* facade)
{
    return nd_attr(facade, "database");
}

PyObject* nd_entity_id_value(PyObject* instance)
{
    PyObject* entity_id = nd_attr(instance, "entity_id");

    if (entity_id == NULL) {
        return NULL;
    }

    PyObject* value = nd_attr(entity_id, "value");

    Py_DECREF(entity_id);

    return value;
}

PyObject* nd_instance_from_id(PyObject* database, int64_t id)
{
    PyObject* entity_id_cls = nd_engine_class(
        "nadata.core.database.entity_id", "EntityId"
    );

    if (entity_id_cls == NULL) {
        return NULL;
    }

    PyObject* args = Py_BuildValue("(K)", (unsigned long long) id);
    PyObject* entity_id = nd_call(entity_id_cls, args, NULL);

    Py_DECREF(args);
    Py_DECREF(entity_id_cls);

    if (entity_id == NULL) {
        return NULL;
    }

    PyObject* get_by_id = nd_attr(database, "get_by_id");

    if (get_by_id == NULL) {
        Py_DECREF(entity_id);
        return NULL;
    }

    PyObject* call_args = PyTuple_Pack(1, entity_id);

    Py_DECREF(entity_id);

    if (call_args == NULL) {
        Py_DECREF(get_by_id);
        return NULL;
    }

    PyObject* instance = nd_call(get_by_id, call_args, NULL);

    Py_DECREF(call_args);

    /* get_by_id returns None for unknown ids -> map to a clear error. */
    if (instance == Py_None) {
        Py_DECREF(instance);
        nd_set_error("no instance for entity id %lld",
                     (long long) id);
        return NULL;
    }

    return instance;
}

/* ------------------------------------------------------------------ */
/* INPUT: NdValue -> PyObject (borrowed input, fresh output)           */
/* ------------------------------------------------------------------ */

static NdStatusCode py_from_array(PyObject* database, const NdValue* value,
                                  PyObject** out)
{
    PyObject* list = PyList_New(0);

    if (list == NULL) {
        return nd_take_pyerror();
    }

    for (int64_t i = 0; i < value->as.array.count; i++) {
        PyObject* item = NULL;
        NdStatusCode status = nd_py_from_value(database,
                                               &value->as.array.items[i],
                                               &item);

        if (status != ND_OK) {
            Py_DECREF(list);
            return status;
        }

        if (PyList_Append(list, item) != 0) {
            Py_DECREF(item);
            Py_DECREF(list);
            return nd_take_pyerror();
        }

        Py_DECREF(item);
    }

    *out = list;

    return ND_OK;
}

static NdStatusCode py_from_dict(PyObject* database, const NdValue* value,
                                 PyObject** out)
{
    PyObject* dict = PyDict_New();

    if (dict == NULL) {
        return nd_take_pyerror();
    }

    for (int64_t i = 0; i < value->as.dict.count; i++) {
        const char* key = value->as.dict.keys[i];
        PyObject* key_obj = PyUnicode_FromString(key);

        if (key_obj == NULL) {
            Py_DECREF(dict);
            return nd_take_pyerror();
        }

        PyObject* item = NULL;
        NdStatusCode status = nd_py_from_value(database,
                                               &value->as.dict.values[i],
                                               &item);

        if (status != ND_OK) {
            Py_DECREF(key_obj);
            Py_DECREF(dict);
            return status;
        }

        if (PyDict_SetItem(dict, key_obj, item) != 0) {
            Py_DECREF(item);
            Py_DECREF(key_obj);
            Py_DECREF(dict);
            return nd_take_pyerror();
        }

        Py_DECREF(item);
        Py_DECREF(key_obj);
    }

    *out = dict;

    return ND_OK;
}

NdStatusCode nd_py_from_value(PyObject* database, const NdValue* value,
                              PyObject** out)
{
    switch (value->tag) {
        case ND_NULL:
            *out = Py_NewRef(Py_None);
            return ND_OK;

        case ND_BOOL:
            *out = PyBool_FromLong(value->as.boolean);
            return *out != NULL ? ND_OK : nd_take_pyerror();

        case ND_INT64:
            *out = PyLong_FromLongLong(value->as.i64);
            return *out != NULL ? ND_OK : nd_take_pyerror();

        case ND_FLOAT64:
            *out = PyFloat_FromDouble(value->as.f64);
            return *out != NULL ? ND_OK : nd_take_pyerror();

        case ND_STR:
            if (value->as.str == NULL) {
                nd_set_error("ND_STR value with NULL payload");
                return ND_ERR_INVALID_ARG;
            }

            *out = PyUnicode_FromString(value->as.str);
            return *out != NULL ? ND_OK : nd_take_pyerror();

        case ND_ARRAY:
            return py_from_array(database, value, out);

        case ND_DICT:
            return py_from_dict(database, value, out);

        case ND_INSTANCE:
            if (database == NULL) {
                nd_set_error("ND_INSTANCE value without a live database");
                return ND_ERR_INVALID_ARG;
            }

            *out = nd_instance_from_id(database, value->as.instance_id);

            return *out != NULL ? ND_OK : nd_take_pyerror();

        default:
            nd_set_error("unknown NdValue tag %d", (int) value->tag);
            return ND_ERR_INVALID_ARG;
    }
}

NdStatusCode nd_py_from_args(PyObject* database,
                             const NdValue* args, const NdValue* kwargs,
                             PyObject** out_args, PyObject** out_kwargs)
{
    *out_args = NULL;
    *out_kwargs = NULL;

    /* Positional: NULL -> empty tuple; else expect ND_ARRAY. */
    if (args == NULL || args->tag == ND_NULL) {
        *out_args = PyTuple_New(0);
    } else if (args->tag != ND_ARRAY) {
        nd_set_error("positional arguments must be an ND_ARRAY");
        return ND_ERR_INVALID_ARG;
    } else {
        NdStatusCode status = nd_py_from_value(database, args, out_args);

        if (status != ND_OK) {
            return status;
        }

        /* nd_py_from_value returns a list; calls accept sequences. */
    }

    /* Keyword: NULL or ND_NULL -> none; else expect ND_DICT. */
    if (kwargs == NULL || kwargs->tag == ND_NULL) {
        return ND_OK;
    }

    if (kwargs->tag != ND_DICT) {
        Py_CLEAR(*out_args);
        nd_set_error("keyword arguments must be an ND_DICT");
        return ND_ERR_INVALID_ARG;
    }

    NdStatusCode status = nd_py_from_value(database, kwargs, out_kwargs);

    if (status != ND_OK) {
        Py_CLEAR(*out_args);
        return status;
    }

    return ND_OK;
}

/* ------------------------------------------------------------------ */
/* OUTPUT: PyObject -> NdValue (caller owns the result)                */
/* ------------------------------------------------------------------ */

static int is_home_type(PyObject* obj)
{
    PyObject* type = PyObject_Type(obj);

    if (type == NULL) {
        PyErr_Clear();
        return 0;
    }

    PyObject* module = nd_attr(type, "__module__");

    Py_DECREF(type);

    if (module == NULL) {
        PyErr_Clear();
        return 0;
    }

    const char* name = PyUnicode_AsUTF8(module);

    Py_DECREF(module);

    if (name == NULL) {
        PyErr_Clear();
        return 0;
    }

    return strncmp(name, ND_HOME_TYPES_MODULE_PREFIX,
                   sizeof(ND_HOME_TYPES_MODULE_PREFIX) - 1) == 0;
}

static NdStatusCode value_from_dict(PyObject* obj, NdValue* out);

static NdStatusCode value_from_array(PyObject* obj, NdValue* out)
{
    Py_ssize_t count = PySequence_Size(obj);

    if (count < 0) {
        return nd_take_pyerror();
    }

    out->tag = ND_ARRAY;
    out->as.array.count = (int64_t) count;
    out->as.array.items = NULL;

    if (count == 0) {
        return ND_OK;
    }

    out->as.array.items = calloc((size_t) count, sizeof(NdValue));

    if (out->as.array.items == NULL) {
        out->as.array.count = 0;
        nd_set_error("out of memory converting array result");
        return ND_ERR_MEMORY;
    }

    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject* item = PySequence_GetItem(obj, i);

        if (item == NULL) {
            nd_free_value(out);
            memset(out, 0, sizeof(*out));
            return nd_take_pyerror();
        }

        NdStatusCode status = nd_value_from_py(item,
                                               &out->as.array.items[i]);

        Py_DECREF(item);

        if (status != ND_OK) {
            nd_free_value(out);
            memset(out, 0, sizeof(*out));
            return status;
        }
    }

    return ND_OK;
}

static NdStatusCode value_from_dict(PyObject* obj, NdValue* out)
{
    Py_ssize_t count = PyDict_Size(obj);

    out->tag = ND_DICT;
    out->as.dict.count = (int64_t) count;
    out->as.dict.keys = NULL;
    out->as.dict.values = NULL;

    if (count == 0) {
        return ND_OK;
    }

    out->as.dict.keys = calloc((size_t) count, sizeof(char*));
    out->as.dict.values = calloc((size_t) count, sizeof(NdValue));

    if (out->as.dict.keys == NULL || out->as.dict.values == NULL) {
        free(out->as.dict.keys);
        free(out->as.dict.values);
        out->as.dict.keys = NULL;
        out->as.dict.values = NULL;
        out->as.dict.count = 0;
        nd_set_error("out of memory converting dict result");
        return ND_ERR_MEMORY;
    }

    Py_ssize_t position = 0;
    PyObject* key = NULL;
    PyObject* item = NULL;
    int64_t written = 0;

    while (PyDict_Next(obj, &position, &key, &item)) {
        if (!PyUnicode_Check(key)) {
            nd_free_value(out);
            memset(out, 0, sizeof(*out));
            nd_set_error("dict result with a non-string key");
            return ND_ERR_TYPE;
        }

        const char* text = PyUnicode_AsUTF8(key);

        if (text == NULL) {
            PyErr_Clear();
            nd_free_value(out);
            memset(out, 0, sizeof(*out));
            nd_set_error("dict result key is not valid UTF-8");
            return ND_ERR_TYPE;
        }

        char* key_copy = nd_strdup(text);

        if (key_copy == NULL) {
            nd_free_value(out);
            memset(out, 0, sizeof(*out));
            nd_set_error("out of memory converting dict result");
            return ND_ERR_MEMORY;
        }

        NdStatusCode status = nd_value_from_py(item,
                                               &out->as.dict.values[written]);

        if (status != ND_OK) {
            free(key_copy);
            nd_free_value(out);
            memset(out, 0, sizeof(*out));
            return status;
        }

        out->as.dict.keys[written] = key_copy;
        written += 1;
    }

    out->as.dict.count = written;

    return ND_OK;
}

NdStatusCode nd_value_from_py(PyObject* obj, NdValue* out)
{
    memset(out, 0, sizeof(*out));

    if (obj == Py_None) {
        out->tag = ND_NULL;
        return ND_OK;
    }

    if (PyBool_Check(obj)) {
        out->tag = ND_BOOL;
        out->as.boolean = (obj == Py_True);
        return ND_OK;
    }

    if (PyLong_Check(obj)) {
        int overflow = 0;
        long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);

        if (overflow != 0) {
            PyErr_Clear();
            nd_set_error("integer result does not fit in 64 bits");
            return ND_ERR_TYPE;
        }

        if (value == -1 && PyErr_Occurred()) {
            return nd_take_pyerror();
        }

        out->tag = ND_INT64;
        out->as.i64 = (int64_t) value;

        return ND_OK;
    }

    if (PyFloat_Check(obj)) {
        out->tag = ND_FLOAT64;
        out->as.f64 = PyFloat_AsDouble(obj);
        return ND_OK;
    }

    if (PyUnicode_Check(obj)) {
        const char* text = PyUnicode_AsUTF8(obj);

        if (text == NULL) {
            return nd_take_pyerror();
        }

        char* copy = nd_strdup(text);

        if (copy == NULL) {
            nd_set_error("out of memory converting string result");
            return ND_ERR_MEMORY;
        }

        out->tag = ND_STR;
        out->as.str = copy;

        return ND_OK;
    }

    if (PyDict_Check(obj)) {
        return value_from_dict(obj, out);
    }

    if (PyList_Check(obj) || PyTuple_Check(obj)) {
        return value_from_array(obj, out);
    }

    /* Home types (DateTime, Color, ...) are not representable in v1. */
    if (is_home_type(obj)) {
        nd_set_error("result of home type '%s' is not representable in "
                     "ABI v1 (see HOME_TYPES_NOT_REPRESENTABLE)",
                     Py_TYPE(obj)->tp_name);

        return ND_ERR_TYPE;
    }

    /* Engine instances carry a 64-bit entity id — pass by identity. */
    if (PyObject_HasAttrString(obj, "entity_id")) {
        PyObject* id_value = nd_entity_id_value(obj);

        if (id_value == NULL) {
            return nd_take_pyerror();
        }

        long long id = PyLong_AsLongLong(id_value);

        Py_DECREF(id_value);

        if (id == -1 && PyErr_Occurred()) {
            return nd_take_pyerror();
        }

        out->tag = ND_INSTANCE;
        out->as.instance_id = (int64_t) id;

        return ND_OK;
    }

    nd_set_error("result of type '%s' cannot cross the ABI boundary",
                 Py_TYPE(obj)->tp_name);

    return ND_ERR_TYPE;
}

/* ------------------------------------------------------------------ */
/* Output memory release                                               */
/* ------------------------------------------------------------------ */

ND_API void nd_free_value(NdValue* value)
{
    if (value == NULL) {
        return;
    }

    switch (value->tag) {
        case ND_STR:
            free(value->as.str);
            break;

        case ND_ARRAY:
            for (int64_t i = 0; i < value->as.array.count; i++) {
                nd_free_value(&value->as.array.items[i]);
            }
            free(value->as.array.items);
            break;

        case ND_DICT:
            for (int64_t i = 0; i < value->as.dict.count; i++) {
                free(value->as.dict.keys[i]);
                nd_free_value(&value->as.dict.values[i]);
            }
            free(value->as.dict.keys);
            free(value->as.dict.values);
            break;

        default:
            break;
    }

    value->tag = ND_NULL;
}

ND_API void nd_free_str(char* str)
{
    free(str);
}
