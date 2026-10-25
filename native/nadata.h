/*
 * nadata.h — Stable C ABI for the Nadata engine.
 *
 * This header is the single source of truth for every native binding
 * (C/C++ link, Kotlin/Java via JNA or Panama, Rust via bindgen, ...).
 * It maps 1:1 onto the Python facade `nadata.api.Nadata` (Choice A —
 * direct facade mapping) and is deliberately shaped so the richer
 * "Choice B" API can be added later without breaking anything (see
 * "Reserved for Choice B" at the bottom).
 *
 * Concurrency contract:
 *   - The shim owns ALL CPython/GIL handling. Native callers may call
 *     any nd_* function from any thread; the shim attaches/detaches
 *     the thread to the interpreter around every call. Host code must
 *     never touch the GIL or Python headers.
 *   - The database is *externally serialized*: concurrent calls are
 *     safe but execute one at a time.
 *
 * Memory contract:
 *   - INPUT values (const NdValue*) are borrowed: the shim copies what
 *     it needs during the call and never keeps references. The caller
 *     keeps ownership and may free/reuse them immediately after.
 *   - OUTPUT values (NdValue* out params, returned char*) are owned by
 *     the CALLER and must be released with nd_free_value() /
 *     nd_free_str(). nd_free_value() is recursive: it frees nested
 *     strings, arrays and dicts.
 *   - Handles (NdDb*, NdQuery*) are released with nd_close() /
 *     nd_query_free(). Using a handle after release is undefined.
 *
 * Strings:
 *   - All strings are UTF-8, NUL-terminated char*. Embedded NUL bytes
 *     are not representable in this ABI version (documented limitation;
 *     a length-paired NdStr variant is reserved for ABI 0.2).
 *
 * Errors:
 *   - Every fallible function returns NdStatus. ND_OK (0) = success;
 *     anything negative is an error class (see NdStatusCode). The
 *     human-readable message for the MOST RECENT call on the current
 *     thread is available via nd_last_error(). The thread-local error
 *     is reset at the start of every call.
 */

#ifndef NADATA_ABI_H
#define NADATA_ABI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Export macro: when building the shim this must be defined as
 * ND_BUILDING_ABI so symbols are exported (dllexport / default
 * visibility). Consumers just include the header.
 */
/*
 * Calling convention: the platform default (cdecl), which on x64
 * Windows is the single defined convention anyway. Bindings generated
 * from this header (JNA, bindgen, ...) must use the default
 * convention — no __stdcall anywhere.
 */
#if defined(_WIN32)
#  if defined(ND_BUILDING_ABI)
#    define ND_EXPORT __declspec(dllexport)
#  else
#    define ND_EXPORT __declspec(dllimport)
#  endif
#else
#  define ND_EXPORT __attribute__((visibility("default")))
#endif

#define ND_API ND_EXPORT

/* ------------------------------------------------------------------ */
/* ABI version                                                         */
/* ------------------------------------------------------------------ */

/*
 * Version of THIS header. Rules:
 *   - Symbols are never removed and signatures never change.
 *   - New symbols are appended; new enum values get new numeric slots.
 *   - Bump when symbols are added so hosts can feature-detect.
 */
#define ND_ABI_VERSION 2

/* ------------------------------------------------------------------ */
/* Status codes                                                        */
/* ------------------------------------------------------------------ */

typedef enum NdStatusCode {
    ND_OK             =  0,  /* success                                        */

    /* Error classes — each maps 1:1 to a Python exception family.     */
    ND_ERR_NOTFOUND   = -1,  /* KeyError        — unknown struct name          */
    ND_ERR_NAME       = -2,  /* NameError       — unresolvable identifier
                                                     in executed source       */
    ND_ERR_VALUE      = -3,  /* ValueError      — bad .ndb, missing file, ...  */
    ND_ERR_TYPE       = -4,  /* TypeError       — argument/attribute mismatch  */
    ND_ERR_SYNTAX     = -5,  /* SyntaxError     — schema resolve-time errors   */
    ND_ERR_IO         = -6,  /* OSError family  — filesystem failures          */
    ND_ERR_LIMIT      = -7,  /* LoopLimitError  — per-loop fuel cap exceeded   */
    ND_ERR_PERMISSION = -8,  /* PermissionError — sandboxed I/O traversal      */
    ND_ERR_RUNTIME    = -9,  /* RuntimeError    — engine failure at run time   */

    /* Shim-level errors (raised before Python is involved).           */
    ND_ERR_INVALID_ARG = -10, /* NULL handle, NULL required string, bad tag */
    ND_ERR_STATE       = -11, /* handle used after free, wrong op for the
                                 way the db was opened                      */
    ND_ERR_MEMORY      = -12  /* allocation failure inside the shim         */
} NdStatusCode;

/*
 * Runtime information, JSON-encoded, e.g.:
 *   {"abi":1,"cpython":"3.12.7","free_threaded":false,"build":"pbs-20241016"}
 * Caller frees with nd_free_str(). Never fails; returns "null" on
 * internal error.
 */
ND_API char* nd_runtime_info(void);

/* ------------------------------------------------------------------ */
/* Values                                                              */
/* ------------------------------------------------------------------ */

typedef enum NdType {
    ND_NULL     = 0,
    ND_BOOL     = 1,
    ND_INT64    = 2,
    ND_FLOAT64  = 3,
    ND_STR      = 4,
    ND_ARRAY    = 5,
    ND_DICT     = 6,
    ND_INSTANCE = 7   /* payload = entity id (int64), stable across calls */
} NdType;

typedef struct NdValue NdValue;

struct NdValue {
    NdType tag;
    union {
        int32_t boolean;                    /* ND_BOOL: 0 or 1           */
        int64_t i64;                        /* ND_INT64                  */
        double f64;                         /* ND_FLOAT64                */
        char* str;                          /* ND_STR: output, nd_free_str
                                               INPUT: caller-owned const  */
        struct {                            /* ND_ARRAY                  */
            NdValue* items;                 /*   heap array of count     */
            int64_t count;
        } array;
        struct {                            /* ND_DICT                   */
            char** keys;                    /*   heap array of count     */
            NdValue* values;                /*   heap array of count     */
            int64_t count;
        } dict;
        int64_t instance_id;                /* ND_INSTANCE               */
    } as;
};

/*
 * Frees an OUTPUT NdValue recursively (strings, arrays, dicts).
 * Passing NULL is a no-op. Do NOT call this on input values you own.
 */
ND_API void nd_free_value(NdValue* value);

/* Frees an OUTPUT string (returned by nd_summary, nd_runtime_info,
 * dict keys inside freed NdValues, ...). Passing NULL is a no-op. */
ND_API void nd_free_str(char* str);

/* ------------------------------------------------------------------ */
/* Database handle — maps to nadata.api.Nadata                         */
/* ------------------------------------------------------------------ */

typedef struct NdDb NdDb;

/*
 * Loads a project folder, a .nastruct file, or an .ndb database
 * (auto-detected), exactly like Nadata.load(path).
 * On failure returns a negative status and sets the thread-local error.
 */
ND_API NdStatusCode nd_load(const char* path, NdDb** out_db);

/* Opens an existing .ndb file, exactly like Nadata.open(path). */
ND_API NdStatusCode nd_open(const char* storage_path, NdDb** out_db);

/*
 * Folder session pipeline, exactly like Nadata.open_folder(...).
 * db_path may be NULL (default '<folder>/nadata.ndb').
 * run_auto / force_init: 0 = false, nonzero = true.
 */
ND_API NdStatusCode nd_open_folder(const char* folder_path,
                                   const char* db_path,       /* nullable */
                                   int32_t run_auto,
                                   int32_t force_init,
                                   NdDb** out_db);

/* Closes the database and frees the handle. NULL is a no-op. */
ND_API void nd_close(NdDb* db);

/* ------------------------- Introspection -------------------------- */

/* All return ND_ARRAY of ND_STR, or a negative status. */
ND_API NdStatusCode nd_struct_names(NdDb* db, NdValue* out);
ND_API NdStatusCode nd_function_names(NdDb* db, NdValue* out);
ND_API NdStatusCode nd_symbol_names(NdDb* db, NdValue* out);

/* Human-readable overview (the `nadata info` text). Caller frees. */
ND_API NdStatusCode nd_summary(NdDb* db, char** out);

/* Polymorphic instance count (struct + subclasses). */
ND_API NdStatusCode nd_count(NdDb* db, const char* struct_name, int64_t* out);

ND_API NdStatusCode nd_total_instance_count(NdDb* db, int64_t* out);

/* -------------------- Instances and attributes -------------------- */

/*
 * Resolves a bound data symbol to its entity id (Nadata.get).
 * Empty/unknown symbol -> ND_ERR_NAME.
 */
ND_API NdStatusCode nd_get_symbol_id(NdDb* db, const char* symbol,
                                     int64_t* out_id);

/*
 * Nadata.get_attribute(symbol, name) — plain unwrapped value out.
 * Unknown symbol -> ND_ERR_NAME; unknown attribute -> ND_ERR_TYPE.
 */
ND_API NdStatusCode nd_get_attribute(NdDb* db, const char* symbol,
                                     const char* name, NdValue* out);

/* Nadata.set_attribute(symbol, name, value). Value is borrowed. */
ND_API NdStatusCode nd_set_attribute(NdDb* db, const char* symbol,
                                     const char* name,
                                     const NdValue* value);

/*
 * Nadata.create_instance(struct_name, attributes, symbol).
 *   attributes: ND_DICT (str keys -> values) or ND_NULL for none.
 *   symbol:     NULL or empty = do not bind.
 * Returns the new instance's entity id.
 * Unknown struct -> ND_ERR_NOTFOUND.
 */
ND_API NdStatusCode nd_create_instance(NdDb* db, const char* struct_name,
                                       const NdValue* attributes, /* nullable */
                                       const char* symbol,        /* nullable */
                                       int64_t* out_id);

/*
 * Nadata.call_function(name, *args, **kwargs).
 *   args:  ND_ARRAY of positional values (borrowed).
 *   kwargs: ND_DICT of str -> value, or ND_NULL.
 * The result (if any) is written to *out; functions returning None
 * write ND_NULL. Call failures map to their Python error class.
 */
ND_API NdStatusCode nd_call_function(NdDb* db, const char* name,
                                     const NdValue* args,   /* nullable */
                                     const NdValue* kwargs, /* nullable */
                                     NdValue* out);

/* Nadata.call_method(symbol, name, *args, **kwargs). Same rules. */
ND_API NdStatusCode nd_call_method(NdDb* db, const char* symbol,
                                   const char* name,
                                   const NdValue* args,   /* nullable */
                                   const NdValue* kwargs, /* nullable */
                                   NdValue* out);

/* ----------------------------- Queries ---------------------------- */

/*
 * QuerySet handle. Nadata's QuerySet methods return NEW QuerySet
 * objects; this ABI keeps ONE handle and rebinds it in place, so
 * nd_query_order_by(...) mutates *qs. Chain freely, then read out.
 */
typedef struct NdQuery NdQuery;

/*
 * Predicate callback for nd_query_where_cb.
 * Return 1 = keep, 0 = drop. Returning a negative NdStatusCode aborts
 * the whole filter with that status. Called on the calling thread with
 * the GIL attached; must not call back into the same db handle
 * (deadlock risk — document for v1).
 */
typedef int32_t (*NdPredicateFn)(int64_t instance_id, void* user_data);

/* Nadata.query(struct_name). Unknown struct -> ND_ERR_NOTFOUND. */
ND_API NdStatusCode nd_query_new(NdDb* db, const char* struct_name,
                                 NdQuery** out_qs);

/*
 * Universal filter: predicate written in the nadata language, evaluated
 * by the engine. Works from EVERY binding, including WASM. This is the
 * primary documented filter style. Example:
 *   nd_query_where_str(qs, "h", "h.age(DateTime.now()).year >= 18");
 */
ND_API NdStatusCode nd_query_where_str(NdQuery* qs, const char* var,
                                       const char* expr);

/* Native-callback filter (C, C++, Rust, JVM). Not available on WASM. */
ND_API NdStatusCode nd_query_where_cb(NdQuery* qs, NdPredicateFn fn,
                                      void* user_data);

ND_API NdStatusCode nd_query_order_by(NdQuery* qs, const char* attribute,
                                      int32_t descending);

/* Invalid (<= 0) n is an ND_ERR_INVALID_ARG. */
ND_API NdStatusCode nd_query_limit(NdQuery* qs, int64_t n);

ND_API NdStatusCode nd_query_count(NdQuery* qs, int64_t* out);
ND_API NdStatusCode nd_query_exists(NdQuery* qs, int32_t* out); /* 0/1 */

/*
 * First/last matching instance id. EMPTY SET IS NOT AN ERROR-FREE CASE:
 * returns ND_ERR_NOTFOUND (explicit, catchable) with out_id untouched —
 * the ABI equivalent of Python's Optional[StructInstance].
 */
ND_API NdStatusCode nd_query_first_id(NdQuery* qs, int64_t* out_id);
ND_API NdStatusCode nd_query_last_id(NdQuery* qs, int64_t* out_id);

/*
 * Raw payloads for one attribute, in instance order
 * (QuerySet.pluck) — ND_ARRAY out.
 */
ND_API NdStatusCode nd_query_pluck(NdQuery* qs, const char* attribute,
                                   NdValue* out);

/* Aggregations. sum/min/max write the natural numeric type; average
 * writes ND_FLOAT64. Non-numeric attribute -> ND_ERR_TYPE. */
ND_API NdStatusCode nd_query_sum_of(NdQuery* qs, const char* attribute,
                                    NdValue* out);
ND_API NdStatusCode nd_query_min_of(NdQuery* qs, const char* attribute,
                                    NdValue* out);
ND_API NdStatusCode nd_query_max_of(NdQuery* qs, const char* attribute,
                                    NdValue* out);
ND_API NdStatusCode nd_query_average_of(NdQuery* qs, const char* attribute,
                                        double* out);

/*
 * All matching ids in instance order (QuerySet.to_list) — ND_ARRAY of
 * ND_INSTANCE out. This is how host languages enumerate results.
 */
ND_API NdStatusCode nd_query_to_list_ids(NdQuery* qs, NdValue* out);

ND_API void nd_query_free(NdQuery* qs);   /* NULL is a no-op */

/* --------------------------- Persistence -------------------------- */

/* Full snapshot; the file must not exist (ValueError -> ND_ERR_VALUE). */
ND_API NdStatusCode nd_dump(NdDb* db, const char* storage_path);

/*
 * Incremental delta append (Nadata.save_changes). *out_docs = number of
 * delta documents appended. Missing file -> ND_ERR_VALUE (dump first).
 */
ND_API NdStatusCode nd_save_changes(NdDb* db, const char* storage_path,
                                    int64_t* out_docs);

/*
 * Rewrite the file to its live state (Nadata.compact).
 *   output_path NULL = compact in place (db is reloaded afterwards).
 *   *out_before / *out_after = document counts, per the Python API.
 * No known target file -> ND_ERR_VALUE.
 */
ND_API NdStatusCode nd_compact(NdDb* db, const char* output_path, /* nullable */
                               int64_t* out_before, int64_t* out_after);

/*
 * Raises the per-loop fuel cap (Nadata.set_loop_limit); persisted in
 * the database settings. n < 1 -> ND_ERR_VALUE (validated by the
 * engine, not the shim).
 */
ND_API NdStatusCode nd_set_loop_limit(NdDb* db, int64_t limit);

/* -------------------------- Code execution ------------------------ */

/* Parse + merge .nastruct source into the LIVE database. */
ND_API NdStatusCode nd_execute_schema(NdDb* db, const char* schema_text);

/* Parse + execute .nadata source (instances, bindings, scripts). */
ND_API NdStatusCode nd_execute_data(NdDb* db, const char* data_text);

/* Auto-detects schema vs data (Nadata.execute). */
ND_API NdStatusCode nd_execute(NdDb* db, const char* text);

/*
 * Runs one #manual script file with its parameter prelude
 * (Nadata.execute_script_file). params: ND_DICT or ND_NULL.
 */
ND_API NdStatusCode nd_execute_script_file(NdDb* db, const char* file_path,
                                           const NdValue* params); /* nullable */

/* ----------------------------- Errors ----------------------------- */

/*
 * Message for the most recent call ON THE CURRENT THREAD (NULL db is
 * allowed — load-time errors are thread-local too). Returns NULL when
 * the last call succeeded. The pointer is valid until the next nd_*
 * call on this thread; copy it if you need to keep it.
 */
ND_API const char* nd_last_error(NdDb* db);

/* ------------------------------------------------------------------ */
/* Choice B — by-entity-id API (added in ABI 2)                        */
/* ------------------------------------------------------------------ */
/*
 * The facade-mapping surface works through SYMBOL names; this layer
 * works through the 64-bit ENTITY IDS returned by nd_get_symbol_id,
 * nd_query_first_id, nd_query_to_list_ids, ... — no symbol lookup,
 * and a coherent "handle = object reference" model for bindings.
 * Append-only: no symbol above changed (hence ND_ABI_VERSION 2).
 */

/* Nadata.get_attribute / set_attribute, addressed by entity id.
 * Unknown id -> ND_ERR_NOTFOUND; unknown attribute -> ND_ERR_NOTFOUND
 * (engine convention); home-type payloads -> ND_ERR_TYPE as usual. */
ND_API NdStatusCode nd_get_attr_by_id(NdDb* db, int64_t id,
                                      const char* name, NdValue* out);

ND_API NdStatusCode nd_set_attr_by_id(NdDb* db, int64_t id,
                                      const char* name,
                                      const NdValue* value);

/* Nadata.call_method addressed by entity id (same result rules as
 * nd_call_method). */
ND_API NdStatusCode nd_call_method_by_id(NdDb* db, int64_t id,
                                         const char* name,
                                         const NdValue* args,
                                         const NdValue* kwargs,
                                         NdValue* out);

/* Instance introspection: ND_ARRAY of ND_STR (schema-declared names). */
ND_API NdStatusCode nd_list_attr(NdDb* db, int64_t id, NdValue* out);
ND_API NdStatusCode nd_list_methods(NdDb* db, int64_t id, NdValue* out);

/* 1 when a live instance exists for the id, 0 otherwise.
 * Never errors on unknown ids (that is the point) — only on NULL args. */
ND_API NdStatusCode nd_instance_id_valid(NdDb* db, int64_t id,
                                         int32_t* out);

/* Subclass-aware named-id lookup (Nadata-style nid namespaces are
 * per base struct). Unknown nid -> ND_ERR_NOTFOUND. */
ND_API NdStatusCode nd_get_by_nid(NdDb* db, const char* base_struct,
                                  const char* nid, int64_t* out_id);

/* ------------------------------------------------------------------ */
/* Still reserved (ABI 0.3+) — do NOT use these names yet.             */
/* ------------------------------------------------------------------ */
/*
 * NdStr (len-paired) variants for embedded-NUL-safe strings.
 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NADATA_ABI_H */
