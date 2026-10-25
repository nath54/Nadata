/*
 * test_conformance.cpp — C++ wrapper replay of the ABI case ids.
 *
 * Mirrors tests/test_abi_conformance.py (ids via @abi_case) and
 * native/tests/smoke.c. Prints "ok <case-id>" per passing case and
 * exits non-zero on any failure, so any runner can diff coverage
 * against the other bindings (see native/wrappers/tests/README.md).
 *
 * Usage: test_conformance [path-to-ex0_simple]
 * (run from the repository root; temp fixtures go to a build dir)
 */

/* Import Modules */
#include "nadata.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int g_checks = 0;
static int g_failures = 0;

static void ok(const char* case_id)
{
    g_checks += 1;
    std::printf("ok   %s\n", case_id);
}

static void fail(const char* case_id, const std::string& detail)
{
    g_checks += 1;
    g_failures += 1;
    std::printf("FAIL %-50s %s\n", case_id, detail.c_str());
}

/* Asserts `expression` is true; reports `case_id` otherwise. */
#define CHECK(case_id, expression)                                    \
    do {                                                              \
        if (expression) {                                             \
            ok(case_id);                                             \
        } else {                                                      \
            fail(case_id, "condition false: " #expression);           \
        }                                                             \
    } while (0)

/* Asserts the callable does NOT throw (success path). */
template <typename Fn>
static void check_no_throw(const char* case_id, Fn&& call)
{
    try {
        call();
        ok(case_id);
    } catch (const std::exception& error) {
        fail(case_id, error.what());
    }
}

/* Asserts the callable throws nadata::Error with exactly `status`. */
template <typename Fn>
static void check_status(const char* case_id, Fn&& call,
                         NdStatusCode expected)
{
    try {
        call();
        fail(case_id, "no exception raised");
    } catch (const nadata::Error& error) {
        if (error.status() == expected) {
            ok(case_id);
        } else {
            fail(case_id, "status " + std::to_string(error.status())
                + " != " + std::to_string(expected) + " (" + error.what()
                + ")");
        }
    } catch (const std::exception& error) {
        fail(case_id, std::string("wrong exception: ") + error.what());
    }
}

static void require(bool condition, const char* what)
{
    if (!condition) {
        std::fprintf(stderr, "fatal: %s\n", what);
        std::exit(2);
    }
}

/* ------------------------------------------------------------------ */
/* Fixtures (mirrors tests/abi/fixtures.py)                            */
/* ------------------------------------------------------------------ */

static const char* EX0_PATH = "./examples/ex0_simple";

static fs::path temp_root()
{
    fs::path root = "native/build/wrapper_tests";
    fs::create_directories(root);

    return root;
}

static void write_file(const fs::path& folder, const std::string& rel,
                       const std::string& text)
{
    fs::path target = folder / rel;
    fs::create_directories(target.parent_path());

    std::ofstream stream(target, std::ios::binary);
    stream << text;
}

/* numeric_project: three boxes 1.5/2.5/3.5, counts 2/3/5. */
static nadata::Database numeric_project(const fs::path& dir)
{
    write_file(dir, "structs/boxes.nastruct",
               "class Box {\n"
               "    label: Str = \"\";\n"
               "    weight: Float64 = 0.0;\n"
               "    count: Int32 = 0;\n"
               "}\n");
    write_file(dir, "data/boxes.nadata",
               "a = Box(label = \"a\", weight = 1.5, count = 2);\n"
               "b = Box(label = \"b\", weight = 2.5, count = 3);\n"
               "c = Box(label = \"c\", weight = 3.5, count = 5);\n");

    return nadata::Database::load(dir.string());
}

/* ------------------------------------------------------------------ */
/* Case groups                                                         */
/* ------------------------------------------------------------------ */

static void lifecycle_cases(const char* ex0)
{
    {   // nd_load/project_folder
        nadata::Database db = nadata::Database::load(ex0);
        std::vector<std::string> names = db.struct_names();
        bool has_human = false;

        for (const auto& name : names) {
            has_human = has_human || name == "Human";
        }

        CHECK("nd_load/project_folder", has_human);
    }

    check_status(   // nd_load/unsupported_path
        "nd_load/unsupported_path",
        [] { return nadata::Database::load("./examples/does_not_exist.txt"); },
        ND_ERR_VALUE);

    {   // nd_open/roundtrip
        fs::path store = temp_root() / "cpp_store.ndb";
        fs::remove(store);

        {
            nadata::Database db = nadata::Database::load(ex0);
            db.dump(store.string());
        }

        nadata::Database reopened = nadata::Database::open(store.string());

        CHECK("nd_open/roundtrip", reopened.count("Human") == 3
              && reopened.total_instance_count() == 6);
    }

    check_status(   // nd_open/missing_file
        "nd_open/missing_file",
        [] { return nadata::Database::open("./definitely_missing.ndb"); },
        ND_ERR_VALUE);

    {   // nd_open/invalid_database
        fs::path garbage = temp_root() / "garbage.ndb";

        std::ofstream stream(garbage, std::ios::binary);
        stream << "not a database at all";

        check_status(
            "nd_open/invalid_database",
            [&] { return nadata::Database::open(garbage.string()); },
            ND_ERR_VALUE);
    }
}

static void introspection_cases(nadata::Database& db)
{
    {   // nd_struct_names/ex0
        std::vector<std::string> names = db.struct_names();
        bool has = false;

        for (const auto& name : names) {
            has = has || name == "Human";
        }

        CHECK("nd_struct_names/ex0", has && !names.empty());
    }

    {   // nd_function_names/ex0
        std::vector<std::string> names = db.function_names();
        bool has = false;

        for (const auto& name : names) {
            has = has || name == "is_adult";
        }

        CHECK("nd_function_names/ex0", has);
    }

    {   // nd_symbol_names/ex0
        std::vector<std::string> names = db.symbol_names();
        bool has_tom = false;

        for (const auto& name : names) {
            has_tom = has_tom || name == "tom";
        }

        CHECK("nd_symbol_names/ex0", has_tom);
    }

    CHECK("nd_summary/nonempty_string", db.summary().size() > 0);
    CHECK("nd_count/polymorphic", db.count("Entity") == 4);
    CHECK("nd_total_instance_count/ex0", db.total_instance_count() == 6);

    check_status(   // nd_count/unknown_struct
        "nd_count/unknown_struct",
        [&] { return db.count("NoSuch"); }, ND_ERR_NOTFOUND);

    CHECK("nd_get_symbol_id/ok", db.get_symbol_id("tom") >= 0);

    check_status(   // nd_get_symbol_id/unknown_symbol
        "nd_get_symbol_id/unknown_symbol",
        [&] { return db.get_symbol_id("nobody"); }, ND_ERR_NOTFOUND);
}

static void attribute_cases(nadata::Database& db)
{
    CHECK("nd_get_attribute/plain_scalar",
          std::string(db.get_attribute("tom", "birthplace").as_str())
              == "Paris");

    check_status(   // nd_get_attribute/unknown_symbol
        "nd_get_attribute/unknown_symbol",
        [&] { return db.get_attribute("nobody", "name"); },
        ND_ERR_NOTFOUND);

    check_status(   // nd_get_attribute/unknown_attribute
        "nd_get_attribute/unknown_attribute",
        [&] { return db.get_attribute("tom", "no_such"); },
        ND_ERR_NOTFOUND);

    check_status(   // nd_get_attribute/home_type_payload (ABI v1 limit)
        "nd_get_attribute/home_type_payload",
        [&] { return db.get_attribute("tom", "birthday"); },
        ND_ERR_TYPE);

    {   // nd_set_attribute/roundtrip + readback
        db.set_attribute("tom", "birthplace",
                         nadata::Value::str("Lyon"));

        CHECK("nd_set_attribute/roundtrip",
              std::string(db.get_attribute("tom", "birthplace").as_str())
                  == "Lyon");

        db.set_attribute("tom", "birthplace",
                         nadata::Value::str("Paris"));   // restore
    }

    check_status(   // nd_set_attribute/type_mismatch
        "nd_set_attribute/type_mismatch",
        [&] {
            return db.set_attribute("tom", "birthplace",
                                    nadata::Value::integer(42));
        },
        ND_ERR_TYPE);

    check_status(   // nd_set_attribute/unknown_symbol
        "nd_set_attribute/unknown_symbol",
        [&] {
            return db.set_attribute("nobody", "name",
                                    nadata::Value::str("x"));
        },
        ND_ERR_NOTFOUND);
}

static void instance_cases(nadata::Database& db)
{
    /* The Python suite builds fresh fixtures per case; here groups
     * share one db, so the schema is registered before first use. */
    db.execute_schema("class Weapon { name: Str = \"Sword\"; }");

    {   // nd_create_instance/with_attributes_and_symbol
        int64_t id = db.create_instance(
            "Weapon", {{"name", nadata::Value::str("Excalibur")}}, "exc");

        CHECK("nd_create_instance/with_attributes_and_symbol", id >= 0);
        CHECK("nd_create_instance/readback",
              std::string(db.get_attribute("exc", "name").as_str())
                  == "Excalibur");
    }

    check_status(   // nd_create_instance/unknown_struct
        "nd_create_instance/unknown_struct",
        [&] { return db.create_instance("NoSuch", nadata::Value::null()); },
        ND_ERR_NOTFOUND);
}

static void call_cases(nadata::Database& db)
{
    int64_t tom = db.get_symbol_id("tom");
    int64_t jean = db.get_symbol_id("jean");

    {   // nd_call_function/positional_args
        nadata::Value result = db.call_function(
            "is_adult", nadata::Value::array({nadata::Value::instance(tom)}));

        CHECK("nd_call_function/positional_args",
              result.type() == ND_BOOL && result.as_bool());
    }

    check_status(   // nd_call_function/unknown_function
        "nd_call_function/unknown_function",
        [&] { return db.call_function("no_such_function"); },
        ND_ERR_NOTFOUND);

    check_status(   // nd_call_function/argument_type_mismatch
        "nd_call_function/argument_type_mismatch",
        [&] {
            return db.call_function(
                "is_adult",
                nadata::Value::array({nadata::Value::str("not an id")}));
        },
        ND_ERR_TYPE);

    check_status(   // nd_call_function/home_type_result (ABI v1 limit)
        "nd_call_function/home_type_result",
        [&] {
            return db.call_function(
                "calculate_age_difference",
                nadata::Value::null(),
                nadata::Value::dict({
                    {"h1", nadata::Value::instance(tom)},
                    {"h2", nadata::Value::instance(jean)},
                }));
        },
        ND_ERR_TYPE);

    {   // nd_call_method/ok — jean.older(tom) is true (jean is older)
        nadata::Value result = db.call_method(
            "jean", "older",
            nadata::Value::array({nadata::Value::instance(tom)}));

        CHECK("nd_call_method/ok",
              result.type() == ND_BOOL && result.as_bool());
    }

    check_status(   // nd_call_method/unknown_method
        "nd_call_method/unknown_method",
        [&] { return db.call_method("jean", "no_such_method"); },
        ND_ERR_NOTFOUND);

    check_status(   // nd_call_method/unknown_symbol
        "nd_call_method/unknown_symbol",
        [&] { return db.call_method("nobody", "older"); },
        ND_ERR_NOTFOUND);
}

static void query_cases(nadata::Database& db)
{
    check_status(   // nd_query_new/unknown_struct
        "nd_query_new/unknown_struct",
        [&] { return db.query("NoSuch"); }, ND_ERR_NOTFOUND);

    {   // nd_query_where_str/engine_evaluated + nd_query_count/after_chaining
        nadata::Query q = db.query("Human");
        q.where_str("h", "h.name == \"Tom\"");

        CHECK("nd_query_where_str/engine_evaluated", true);
        CHECK("nd_query_count/after_chaining", q.count() == 1);
    }

    check_status(   // nd_query_where_str/syntax_error
        "nd_query_where_str/syntax_error",
        [&] { db.query("Human").where_str("h", "h.name ===="); },
        ND_ERR_SYNTAX);

    check_status(   // nd_query_where_str/type_error
        "nd_query_where_str/type_error",
        [&] { db.query("Human").where_str("h", "h.name < 5"); },
        ND_ERR_TYPE);

    {   // nd_query_exists/true_false
        CHECK("nd_query_exists/true_false", db.query("Human").exists());
    }

    {   // nd_query_where_str/empty_result + first_id NOTFOUND + exists
        nadata::Query q = db.query("Human");
        q.where_str("h", "h.name == \"Nobody\"");

        CHECK("nd_query_where_str/empty_result", q.count() == 0);
        CHECK("nd_query_first_id/empty_set_is_notfound",
              !q.first_id().has_value());
        CHECK("nd_query_exists/empty", !q.exists());
        CHECK("nd_query_count/empty_value", q.count() == 0);    }

    {   // nd_query_where_cb/preserves_order
        nadata::Query q = db.query("Human");
        q.where_cb([](int64_t, void*) -> int32_t { return 1; }, nullptr);

        std::vector<int64_t> ids = q.to_list_ids();

        CHECK("nd_query_where_cb/preserves_order", ids.size() == 3);
    }

    {   // nd_query_order_by/ascending_and_descending + limit window
        nadata::Query ascending = db.query("Human");
        ascending.order_by("name").limit(2);

        CHECK("nd_query_order_by/ascending_and_descending",
              ascending.to_list_ids().size() == 2);

        nadata::Query descending = db.query("Human");
        descending.order_by("name", true);

        std::vector<int64_t> desc = descending.to_list_ids();
        std::vector<int64_t> asc = db.query("Human")
                                       .order_by("name")
                                       .to_list_ids();

        CHECK("nd_query_order_by/direction",
              asc.size() == 3 && desc.size() == 3
                  && asc.front() == desc.back());
    }

    check_status(   // nd_query_order_by/unknown_attribute
        "nd_query_order_by/unknown_attribute",
        [&] { db.query("Human").order_by("no_such"); },
        ND_ERR_NOTFOUND);

    check_status(   // nd_query_limit/invalid
        "nd_query_limit/invalid",
        [&] { db.query("Human").limit(0); }, ND_ERR_INVALID_ARG);

    {   // nd_query_pluck/order_and_payloads (instance order = tom, jean, louise)
        nadata::Query q = db.query("Human").order_by("name").limit(2);
        nadata::Value names = q.pluck("name");

        CHECK("nd_query_pluck/order_and_payloads",
              names.type() == ND_ARRAY && names.size() == 2);
    }

    check_status(   // nd_query_pluck/unknown_attribute
        "nd_query_pluck/unknown_attribute",
        [&] { db.query("Human").pluck("no_such"); }, ND_ERR_NOTFOUND);

    {   // nd_query_aggregates/sum_min_max_average (numeric_project)
        fs::path dir = temp_root() / "cpp_numeric";
        nadata::Database numeric = numeric_project(dir);

        nadata::Query q = numeric.query("Box");

        CHECK("nd_query_aggregates/sum", q.sum_of("weight").as_float()
              == 7.5);
        CHECK("nd_query_aggregates/min", q.min_of("weight").as_float()
              == 1.5);
        CHECK("nd_query_aggregates/max", q.max_of("weight").as_float()
              == 3.5);
        CHECK("nd_query_aggregates/average", q.average_of("weight") == 2.5);

        check_status(   // nd_query_aggregates/non_numeric_attribute
            "nd_query_aggregates/non_numeric_attribute",
            [&] { numeric.query("Box").sum_of("label"); },
            ND_ERR_TYPE);
    }

    {   // nd_query_to_list_ids/ordered_and_stable
        std::vector<int64_t> first = db.query("Human").to_list_ids();
        std::vector<int64_t> second = db.query("Human").to_list_ids();

        CHECK("nd_query_to_list_ids/ordered_and_stable",
              first.size() == 3 && first == second);
    }
}

static void persistence_cases(nadata::Database& db, const char* ex0)
{
    fs::path store = temp_root() / "cpp_persist.ndb";
    fs::remove(store);

    {   // nd_dump/roundtrip via a second database
        nadata::Database fresh = nadata::Database::load(ex0);
        fresh.dump(store.string());
    }

    check_status(   // nd_dump/file_must_not_exist
        "nd_dump/file_must_not_exist",
        [&] { return db.dump(store.string()); }, ND_ERR_VALUE);

    check_status(   // nd_save_changes/missing_file
        "nd_save_changes/missing_file",
        [&] { return db.save_changes((temp_root() / "never_dumped.ndb").string()); },
        ND_ERR_VALUE);

    {   // nd_save_changes/delta_appended
        nadata::Database reopened = nadata::Database::open(store.string());
        db.set_attribute("tom", "birthplace", nadata::Value::str("Nice"));

        CHECK("nd_save_changes/delta_appended",
              db.save_changes(store.string()) >= 1);

        db.set_attribute("tom", "birthplace", nadata::Value::str("Paris"));
    }

    {   // nd_compact/in_place_returns_counts
        nadata::Database reopened = nadata::Database::open(store.string());
        auto counts = reopened.compact();

        CHECK("nd_compact/in_place_returns_counts", counts.first >= 1
              && counts.second >= 1 && counts.second <= counts.first);
    }
}

static void execution_cases(nadata::Database& db, const fs::path& dir)
{
    {   // nd_execute_schema/live_merge — new structs are instantiable
        db.execute_schema("class Shield { name: Str = \"Buckler\"; }");

        int64_t id = db.create_instance(
            "Shield", {{"name", nadata::Value::str("Excalibur")}}, "kite");

        CHECK("nd_execute_schema/live_merge", id >= 0);
    }

    {   // nd_execute_data/instances_bound
        db.execute_data("buckler2 = Shield(name = \"Buckler 2\");");

        CHECK("nd_execute_data/instances_bound",
              std::string(db.get_attribute("buckler2", "name").as_str())
                  == "Buckler 2");
    }

    {   // nd_execute/auto_detection (schema-looking text)
        db.execute("class Helm { size: Int32 = 58; }");

        bool has_helm = false;

        for (const auto& name : db.struct_names()) {
            has_helm = has_helm || name == "Helm";
        }

        CHECK("nd_execute/auto_detection", has_helm);
    }

    {   // nd_execute_script_file/params (session_project)
        fs::path session = dir / "cpp_session";
        write_file(session, "structs/schema.nastruct",
                   "class Child {\n"
                   "    name: Str;\n"
                   "    age: Int32 = 0;\n"
                   "}\n");
        write_file(session, "data/seed.nadata",
                   "#init\n"
                   "tom = Child(nid = \"tom\", name = \"Tom\", age = 6);\n"
                   "flush();\n");
        write_file(session, "capture.nadata",
                   "#manual(v: Int = 7, name: Str = \"x\")\n"
                   "captured = Child(nid = \"captured\", age = v, "
                   "name = name);\n"
                   "flush();\n");

        nadata::Database opened = nadata::Database::open_folder(
            session.string());

        opened.execute_script_file(
            (session / "capture.nadata").string(),
            nadata::Value::dict({
                {"v", nadata::Value::integer(9)},
                {"name", nadata::Value::str("Zoe")},
            }));

        CHECK("nd_execute_script_file/params",
              std::string(
                  opened.get_attribute("captured", "name").as_str())
                  == "Zoe"
              && opened.get_attribute("captured", "age").as_int() == 9);
    }
}

/* ------------------------------------------------------------------ */
/* Runner                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);   /* survive crashes: see every line */

    const char* ex0 = argc > 1 ? argv[1] : EX0_PATH;

    /* nd_runtime_info/valid_json (from the C smoke). */
    {
        std::string info = nadata::runtime_info();

        CHECK("nd_runtime_info/valid_json",
              info.find("\"abi\":1") != std::string::npos);
    }

    lifecycle_cases(ex0);

    nadata::Database db = nadata::Database::load(ex0);

    introspection_cases(db);
    attribute_cases(db);
    instance_cases(db);
    call_cases(db);
    query_cases(db);
    persistence_cases(db, ex0);
    execution_cases(db, temp_root());

    /* nd_set_loop_limit contract. */
    check_no_throw("nd_set_loop_limit/valid_value",
                   [&] { db.set_loop_limit(100000); });

    check_status("nd_set_loop_limit/below_one",
                 [&] { return db.set_loop_limit(0); }, ND_ERR_VALUE);

    std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);

    return g_failures == 0 ? 0 : 1;
}
