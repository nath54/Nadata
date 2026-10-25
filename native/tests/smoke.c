/*
 * smoke.c — C smoke test for the nadata shim.
 *
 * Replays key conformance case ids (tests/test_abi_conformance.py)
 * through the raw C ABI and exits non-zero on the first failure.
 * Each CHECK prints "ok <case-id>" so failures map straight back to
 * the conformance suite.
 *
 * Usage: smoke [path-to-ex0_simple]
 * (defaults to ./examples/ex0_simple — run from the repository root)
 */

/* Import Modules */
#include "nadata.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static int g_checks = 0;

/* Asserts status == expected; prints the shim error message when not. */
static void check_status(const char* case_id, NdStatusCode status,
                         NdStatusCode expected)
{
    g_checks += 1;

    if (status != expected) {
        g_failures += 1;
        printf("FAIL %-50s got %d, expected %d: %s\n",
               case_id, (int) status, (int) expected, nd_last_error(NULL));
    } else {
        printf("ok   %s\n", case_id);
    }
}

/* Asserts an int64 output. */
static void check_int(const char* case_id, int64_t actual, int64_t expected)
{
    g_checks += 1;

    if (actual != expected) {
        g_failures += 1;
        printf("FAIL %-50s got %lld, expected %lld\n",
               case_id, (long long) actual, (long long) expected);
    } else {
        printf("ok   %s\n", case_id);
    }
}

/* Asserts a string output. */
static void check_str(const char* case_id, const char* actual,
                      const char* expected)
{
    g_checks += 1;

    if (actual == NULL || strcmp(actual, expected) != 0) {
        g_failures += 1;
        printf("FAIL %-50s got %s, expected \"%s\"\n",
               case_id, actual ? actual : "(null)", expected);
    } else {
        printf("ok   %s\n", case_id);
    }
}

static void require(NdStatusCode status, const char* what)
{
    if (status != ND_OK) {
        fprintf(stderr, "fatal: %s failed (%d): %s\n",
                what, (int) status, nd_last_error(NULL));
        exit(2);
    }
}

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);   /* survive crashes: see every line */

    const char* project = argc > 1
        ? argv[1] : "./examples/ex0_simple";

    /* nd_runtime_info: JSON with the documented keys. */
    char* info = nd_runtime_info();

    check_status("nd_runtime_info/valid_json",
                 info != NULL && strstr(info, "\"abi\":1") != NULL
                     ? ND_OK : ND_ERR_RUNTIME, ND_OK);
    printf("     runtime: %s\n", info ? info : "(null)");
    nd_free_str(info);

    /* nd_load / nd_close */
    NdDb* db = NULL;

    check_status("nd_load/project_folder", nd_load(project, &db), ND_OK);
    if (db == NULL) { fprintf(stderr, "fatal: null db handle\n"); return 2; }

    /* nd_struct_names */
    NdValue names;

    check_status("nd_struct_names/ex0", nd_struct_names(db, &names), ND_OK);

    check_status("nd_struct_names/is_array",
                 names.tag == ND_ARRAY ? ND_OK : ND_ERR_TYPE, ND_OK);
    nd_free_value(&names);

    /* nd_count (polymorphic) */

    int64_t entities = -1;

    check_status("nd_count/polymorphic",
                 nd_count(db, "Entity", &entities), ND_OK);

    check_int("nd_count/value", entities, 4);

    int64_t missing = 0;

    check_status("nd_count/unknown_struct",
                 nd_count(db, "NoSuch", &missing), ND_ERR_NOTFOUND);

    /* nd_total_instance_count */
    int64_t total = -1;

    check_status("nd_total_instance_count/ex0",
                 nd_total_instance_count(db, &total), ND_OK);
    check_int("nd_total_instance_count/value", total, 6);

    /* nd_get_symbol_id + nd_get_attribute */

    int64_t tom = -1;

    check_status("nd_get_symbol_id/ok",
                 nd_get_symbol_id(db, "tom", &tom), ND_OK);

    int64_t nobody = 0;

    check_status("nd_get_symbol_id/unknown_symbol",
                 nd_get_symbol_id(db, "nobody", &nobody), ND_ERR_NOTFOUND);

    NdValue birthplace;

    check_status("nd_get_attribute/plain_scalar",
                 nd_get_attribute(db, "tom", "birthplace", &birthplace),
                 ND_OK);

    check_str("nd_get_attribute/value", birthplace.as.str, "Paris");
    nd_free_value(&birthplace);

    check_status("nd_get_attribute/unknown_attribute",
                 nd_get_attribute(db, "tom", "no_such", &birthplace),
                 ND_ERR_NOTFOUND);

    /* nd_set_attribute roundtrip */

    NdValue lyon;

    lyon.tag = ND_STR;
    lyon.as.str = "Lyon";

    check_status("nd_set_attribute/roundtrip",
                 nd_set_attribute(db, "tom", "birthplace", &lyon), ND_OK);

    check_status("nd_set_attribute/readback",
                 nd_get_attribute(db, "tom", "birthplace", &birthplace),
                 ND_OK);
    check_str("nd_set_attribute/value", birthplace.as.str, "Lyon");
    nd_free_value(&birthplace);

    /* nd_call_method: jean.older(tom) is true (jean is older). */
    int64_t jean = -1;

    require(nd_get_symbol_id(db, "jean", &jean), "get jean");

    NdValue older_args;

    older_args.tag = ND_ARRAY;
    older_args.as.array.count = 1;
    older_args.as.array.items = malloc(sizeof(NdValue));
    older_args.as.array.items[0].tag = ND_INSTANCE;
    older_args.as.array.items[0].as.instance_id = tom;

    NdValue older_result;

    check_status("nd_call_method/ok",
                 nd_call_method(db, "jean", "older", &older_args, NULL,
                                &older_result),
                 ND_OK);
    check_int("nd_call_method/value",
              older_result.tag == ND_BOOL &&
                  older_result.as.boolean == 1 ? 1 : 0, 1);
    nd_free_value(&older_result);
    free(older_args.as.array.items);

    /* nd_query_new + where_str + pluck + count (chained) */
    NdQuery* qs = NULL;

    check_status("nd_query_new/ok", nd_query_new(db, "Human", &qs), ND_OK);
    if (qs == NULL) { fprintf(stderr, "fatal: null query handle\n"); return 2; }

    check_status("nd_query_where_str/engine_evaluated",
                 nd_query_where_str(qs, "h", "h.name == \"Tom\""), ND_OK);

    int64_t filtered = -1;

    check_status("nd_query_count/after_chaining",
                 nd_query_count(qs, &filtered), ND_OK);
    check_int("nd_query_count/value", filtered, 1);

    check_status("nd_query_where_str/syntax_error",
                 nd_query_where_str(qs, "h", "h.name ===="),
                 ND_ERR_SYNTAX);

    /* Fresh query: order_by + limit + to_list_ids. */
    require(nd_query_new(db, "Human", &qs), "requery");
    check_status("nd_query_order_by/ascending",
                 nd_query_order_by(qs, "name", 0), ND_OK);
    check_status("nd_query_limit/window", nd_query_limit(qs, 2), ND_OK);

    NdValue ids;

    check_status("nd_query_to_list_ids/ordered",
                 nd_query_to_list_ids(qs, &ids), ND_OK);
    check_status("nd_query_to_list_ids/is_array",
                 ids.tag == ND_ARRAY && ids.as.array.count == 2
                     ? ND_OK : ND_ERR_TYPE, ND_OK);
    nd_free_value(&ids);

    /* Empty set: first_id -> ND_ERR_NOTFOUND (ids are 0-based). */
    require(nd_query_new(db, "Human", &qs), "requery");
    check_status("nd_query_where_str/empty_result",
                 nd_query_where_str(qs, "h", "h.name == \"Nobody\""), ND_OK);

    int64_t first = 0;

    check_status("nd_query_first_id/empty_set_is_notfound",
                 nd_query_first_id(qs, &first), ND_ERR_NOTFOUND);

    int32_t exists = -1;

    check_status("nd_query_exists/empty",
                 nd_query_exists(qs, &exists), ND_OK);
    check_int("nd_query_exists/value", exists, 0);

    int64_t count = -1;

    check_status("nd_query_count/empty", nd_query_count(qs, &count), ND_OK);
    check_int("nd_query_count/empty_value", count, 0);

    nd_query_free(qs);

    /* nd_dump + nd_open roundtrip. */
    const char* store = "native/build/smoke_store.ndb";

    remove(store);
    check_status("nd_dump/roundtrip", nd_dump(db, store), ND_OK);

    NdDb* reopened = NULL;

    check_status("nd_open/roundtrip", nd_open(store, &reopened), ND_OK);
    if (reopened == NULL) { fprintf(stderr, "fatal: null reopened handle\n"); return 2; }

    check_status("nd_dump/file_must_not_exist", nd_dump(db, store),
                 ND_ERR_VALUE);

    int64_t reopened_count = -1;

    check_status("nd_open/count_matches",
                 nd_count(reopened, "Human", &reopened_count), ND_OK);
    check_int("nd_open/value", reopened_count, 3);

    /* nd_execute_schema + create_instance + get_attribute. */
    check_status("nd_execute_schema/live_merge",
                 nd_execute_schema(db,
                     "class Weapon { name: Str = \"Sword\"; }"), ND_OK);

    NdValue attrs;

    attrs.tag = ND_DICT;
    attrs.as.dict.count = 1;
    attrs.as.dict.keys = malloc(sizeof(char*));
    attrs.as.dict.keys[0] = "name";
    attrs.as.dict.values = malloc(sizeof(NdValue));
    attrs.as.dict.values[0].tag = ND_STR;
    attrs.as.dict.values[0].as.str = "Excalibur";

    int64_t exc = -1;

    check_status("nd_create_instance/with_attributes_and_symbol",
                 nd_create_instance(db, "Weapon", &attrs, "exc", &exc),
                 ND_OK);

    free(attrs.as.dict.keys);
    free(attrs.as.dict.values);

    check_status("nd_create_instance/readback",
                 nd_get_attribute(db, "exc", "name", &birthplace), ND_OK);
    check_str("nd_create_instance/value", birthplace.as.str, "Excalibur");
    nd_free_value(&birthplace);

    check_status("nd_create_instance/unknown_struct",
                 nd_create_instance(db, "NoSuch", NULL, NULL, &exc),
                 ND_ERR_NOTFOUND);

    /* nd_set_loop_limit contract. */
    check_status("nd_set_loop_limit/valid_value",
                 nd_set_loop_limit(db, 100000), ND_OK);
    check_status("nd_set_loop_limit/below_one",
                 nd_set_loop_limit(db, 0), ND_ERR_VALUE);

    /* Error message availability after a failure. */
    const char* message = nd_last_error(db);

    check_status("nd_last_error/has_message",
                 message != NULL && strstr(message, "Loop limit") != NULL
                     ? ND_OK : ND_ERR_RUNTIME, ND_OK);

    nd_close(reopened);
    nd_close(db);

    printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);

    return g_failures == 0 ? 0 : 1;
}
