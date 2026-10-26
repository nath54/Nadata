import nadata.Nadata;
import nadata.Nadata.Error;
import nadata.NadataLibrary;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;

/**
 * Java conformance replay -- the same case ids and expectations as
 * native/wrappers/tests/cpp/test_conformance.cpp (which replays the
 * canonical tests/test_abi_conformance.py registry).
 *
 * nd_query_where_cb is intentionally absent (documented in
 * NadataLibrary): JNA callbacks run on attached JVM threads and cannot
 * honour the GIL contract safely.
 *
 * Usage: java Conformance [path-to-ex0_simple]  (from the repo root)
 */
public final class Conformance {

    private static int failures = 0;

    private static void ok(String case_id, boolean condition) {
        if (condition) {
            System.out.println("ok   " + case_id);
        } else {
            System.out.println("FAIL " + case_id);
            failures++;
        }
    }

    private interface Thunk {
        Object run() throws Exception;
    }

    private static void okStatus(String case_id, int expected,
                                 Thunk thunk) {
        try {
            thunk.run();
            System.out.println("FAIL " + case_id + " (no exception)");
            failures++;
        } catch (Error error) {
            if (error.status == expected) {
                System.out.println("ok   " + case_id);
            } else {
                System.out.println("FAIL " + case_id + " status "
                    + error.status + " != " + expected + ": "
                    + error.getMessage());
                failures++;
            }
        } catch (Exception exception) {
            System.out.println("FAIL " + case_id + " unexpected: "
                + exception);
            failures++;
        }
    }

    private static void write_file(Path dir, String rel, String text)
            throws IOException {
        Path target = dir.resolve(rel);
        Files.createDirectories(target.getParent());
        Files.write(target, text.getBytes(StandardCharsets.UTF_8));
    }

    public static void main(String[] args) throws Exception {
        System.setOut(new java.io.PrintStream(System.out, true, "UTF-8"));

        /* Library path: -Dnadata.library=... or the dev build dir. */
        String lib_path = System.getProperty("nadata.library");

        if (lib_path == null) {
            String name = System.getProperty("os.name", "").toLowerCase();
            String dir = name.contains("win") ? "win32"
                : name.contains("mac") ? "darwin" : "linux";
            String file = name.contains("win") ? "nadata.dll"
                : name.contains("mac") ? "libnadata.dylib"
                : "libnadata.so";

            lib_path = Path.of("native", "build", dir, file).toString();
        }

        Nadata.loadLibrary(lib_path);

        final String ex0 = args.length > 0
            ? args[0] : "./examples/ex0_simple";

        /* nd_runtime_info/valid_json */
        String info = Nadata.Database.runtimeInfo();
        ok("nd_runtime_info/valid_json",
            info.contains("\"abi\":2") && info.contains("\"cpython\":"));

        /* nd_load/... */
        okStatus("nd_load/unsupported_path", NadataLibrary.ND_ERR_VALUE,
            () -> { Nadata.Database.load("./nadata/__init__.py"); return null; });

        try (Nadata.Database db = Nadata.Database.load(ex0)) {
            /* nd_load/project_folder */
            System.out.println("ok   nd_load/project_folder");

            /* nd_struct_names/ex0 */
            ok("nd_struct_names/ex0", db.getStructNames().contains("Human"));

            /* nd_function_names/ex0 */
            ok("nd_function_names/ex0",
                db.getFunctionNames().contains("is_adult"));

            /* nd_symbol_names/ex0 */
            ok("nd_symbol_names/ex0", db.getSymbolNames().containsAll(
                Arrays.asList("tom", "jean")));

            /* nd_summary/nonempty_string */
            ok("nd_summary/nonempty_string", !db.getSummary().isEmpty());

            /* nd_count/polymorphic + nd_total_instance_count */
            ok("nd_count/polymorphic", db.getCount("Entity") == 4);
            ok("nd_total_instance_count/ex0",
                db.getTotalInstanceCount() == 6);

            /* nd_count/unknown_struct */
            okStatus("nd_count/unknown_struct", NadataLibrary.ND_ERR_NOTFOUND,
                () -> { db.getCount("Nope"); return null; });

            /* nd_get_symbol_id */
            final long tom = db.getSymbolId("tom");
            ok("nd_get_symbol_id/ok", tom >= 0);

            okStatus("nd_get_symbol_id/unknown_symbol",
                NadataLibrary.ND_ERR_NOTFOUND,
                () -> { db.getSymbolId("nobody"); return null; });

            /* nd_get_attribute */
            ok("nd_get_attribute/plain_scalar",
                "Paris".equals(db.getAttribute("tom", "birthplace")));

            okStatus("nd_get_attribute/unknown_symbol",
                NadataLibrary.ND_ERR_NOTFOUND,
                () -> { db.getAttribute("nobody", "name"); return null; });

            okStatus("nd_get_attribute/unknown_attribute",
                NadataLibrary.ND_ERR_NOTFOUND,
                () -> { db.getAttribute("tom", "no_such"); return null; });

            okStatus("nd_get_attribute/home_type_payload",
                NadataLibrary.ND_ERR_TYPE,
                () -> { db.getAttribute("tom", "birthday"); return null; });

            /* nd_set_attribute */
            db.setAttribute("tom", "birthplace", "Lyon");
            ok("nd_set_attribute/roundtrip",
                "Lyon".equals(db.getAttribute("tom", "birthplace")));
            db.setAttribute("tom", "birthplace", "Paris");

            okStatus("nd_set_attribute/type_mismatch",
                NadataLibrary.ND_ERR_TYPE,
                () -> { db.setAttribute("tom", "birthplace", 12345L); return null; });

            okStatus("nd_set_attribute/unknown_symbol",
                NadataLibrary.ND_ERR_NOTFOUND,
                () -> { db.setAttribute("nobody", "name", "x"); return null; });

            /* nd_create_instance */
            Map<String, Object> attributes = new HashMap<>();
            attributes.put("name", "Shadow");
            long shadow = db.createInstance("Cat", attributes, "shadow");
            ok("nd_create_instance/with_attributes_and_symbol", shadow >= 0);
            ok("nd_create_instance/readback",
                "Shadow".equals(db.getAttribute("shadow", "name")));

            okStatus("nd_create_instance/unknown_struct",
                NadataLibrary.ND_ERR_NOTFOUND,
                () -> { db.createInstance("Nope", null, null); return null; });

            /* nd_call_function */
            Object adult = db.callFunction("is_adult",
                Nadata.instance(tom));
            ok("nd_call_function/positional_args",
                Boolean.TRUE.equals(adult));

            okStatus("nd_call_function/unknown_function",
                NadataLibrary.ND_ERR_NOTFOUND,
                () -> { db.callFunction("no_such_fn"); return null; });

            okStatus("nd_call_function/argument_type_mismatch",
                NadataLibrary.ND_ERR_TYPE,
                () -> { db.callFunction("is_adult", "tom"); return null; });

            okStatus("nd_call_function/home_type_result",
                NadataLibrary.ND_ERR_TYPE,
                () -> {
                    db.callFunction("calculate_age_difference",
                        Nadata.instance(tom), Nadata.instance(tom));
                    return null;
                });

            /* nd_call_method */
            ok("nd_call_method/ok",
                Boolean.TRUE.equals(db.callMethod(
                    "jean", "older", Nadata.instance(tom))));

            okStatus("nd_call_method/unknown_method",
                NadataLibrary.ND_ERR_NOTFOUND,
                () -> { db.callMethod("tom", "no_such_method"); return null; });

            okStatus("nd_call_method/unknown_symbol",
                NadataLibrary.ND_ERR_NOTFOUND,
                () -> { db.callMethod("nobody", "older"); return null; });

            /* nd_query_* */
            okStatus("nd_query_new/unknown_struct",
                NadataLibrary.ND_ERR_NOTFOUND,
                () -> { db.query("Nope"); return null; });

            long chained = db.query("Human")
                .whereStr("h", "h.name == \"Tom\"")
                .count();
            ok("nd_query_where_str/engine_evaluated", chained == 1);

            okStatus("nd_query_where_str/syntax_error",
                NadataLibrary.ND_ERR_SYNTAX,
                () -> { db.query("Human").whereStr("h", "h ===="); return null; });

            okStatus("nd_query_where_str/type_error",
                NadataLibrary.ND_ERR_TYPE,
                () -> { db.query("Human").whereStr("h", "h.name > 5"); return null; });

            try (Nadata.Query empty = db.query("Human")
                    .whereStr("h", "h.name == \"Nobody\"")) {
                ok("nd_query_exists/empty", !empty.exists());
                ok("nd_query_count/empty_value", empty.count() == 0);

                ok("nd_query_first_id/empty_set_is_notfound",
                    empty.firstId().isEmpty());
            }

            ok("nd_query_exists/true_false",
                db.query("Human").exists());

            /* nd_query_order_by � limit window + direction (C++ exact) */
            try (Nadata.Query window = db.query("Human")
                    .orderBy("name", false).limit(2)) {
                ok("nd_query_order_by/ascending_and_descending",
                    window.toListIds().size() == 2);
            }

            List<Long> ascending = db.query("Human")
                .orderBy("name", false).toListIds();
            List<Long> descending = db.query("Human")
                .orderBy("name", true).toListIds();
            ok("nd_query_order_by/direction",
                ascending.size() == 3 && descending.size() == 3
                    && ascending.get(0).equals(
                        descending.get(descending.size() - 1)));

            okStatus("nd_query_order_by/unknown_attribute",
                NadataLibrary.ND_ERR_NOTFOUND,
                () -> { db.query("Human").orderBy("no_such", false); return null; });

            okStatus("nd_query_limit/invalid", NadataLibrary.ND_ERR_INVALID_ARG,
                () -> { db.query("Human").limit(0); return null; });

            /* nd_query_pluck */
            List<Object> plucked = db.query("Human")
                .whereStr("h", "h.name == \"Tom\"")
                .pluck("birthplace");
            ok("nd_query_pluck/order_and_payloads",
                plucked.size() == 1 && "Paris".equals(plucked.get(0)));

            okStatus("nd_query_pluck/unknown_attribute",
                NadataLibrary.ND_ERR_NOTFOUND,
                () -> { db.query("Human").pluck("no_such"); return null; });

            /* nd_query_aggregates -- numeric fixture (numeric_project:
             * 3 Boxes, weights 1.5/2.5/3.5, counts 2/3/5). */
            Path numeric = Files.createTempDirectory("nadata-java-num");
            write_file(numeric, "structs/schema.nastruct",
                "class Box {\n    label: Str = \"\";\n"
                + "    weight: Float64;\n    count: Int32;\n}\n");
            write_file(numeric, "data/seed.nadata",
                "#init\n"
                + "b1 = Box(nid = \"b1\", weight = 1.5, count = 2);\n"
                + "b2 = Box(nid = \"b2\", weight = 2.5, count = 3);\n"
                + "b3 = Box(nid = \"b3\", weight = 3.5, count = 5);\n"
                + "flush();\n");

            try (Nadata.Database boxes = Nadata.Database.load(
                    numeric.toString())) {
                try (Nadata.Query all = boxes.query("Box")) {
                    ok("nd_query_aggregates/sum",
                        Math.abs(all.sumOf("weight") - 7.5) < 1e-6);
                    ok("nd_query_aggregates/min",
                        Math.abs(all.minOf("weight") - 1.5) < 1e-6);
                    ok("nd_query_aggregates/max",
                        Math.abs(all.maxOf("weight") - 3.5) < 1e-6);
                    ok("nd_query_aggregates/average",
                        Math.abs(all.averageOf("weight") - 2.5) < 1e-6);
                }

                okStatus("nd_query_aggregates/non_numeric_attribute",
                    NadataLibrary.ND_ERR_TYPE,
                    () -> { boxes.query("Box").sumOf("label"); return null; });
            }

            /* nd_query_to_list_ids */
            List<Long> ids = db.query("Human").toListIds();
            ok("nd_query_to_list_ids/ordered_and_stable",
                ids.size() == 3 && ids.equals(db.query("Human").toListIds()));

            /* nd_dump/nd_open roundtrip */
            Path store = Files.createTempFile("nadata-java", ".ndb");
            Files.deleteIfExists(store);

            try (Nadata.Database fresh = Nadata.Database.load(ex0)) {
                fresh.dump(store.toString());
                System.out.println("ok   nd_dump/roundtrip");

                okStatus("nd_dump/file_must_not_exist",
                    NadataLibrary.ND_ERR_VALUE,
                    () -> { fresh.dump(store.toString()); return null; });

                try (Nadata.Database reopened = Nadata.Database.open(
                        store.toString())) {
                    ok("nd_open/roundtrip",
                        reopened.getCount("Human") == 3);
                }

                okStatus("nd_open/missing_file", NadataLibrary.ND_ERR_VALUE,
                    () -> { Nadata.Database.open(store + ".missing"); return null; });
            }

            /* nd_save_changes */
            okStatus("nd_save_changes/missing_file", NadataLibrary.ND_ERR_VALUE,
                () -> { db.saveChanges(store + ".missing"); return null; });

            Path delta_store = Files.createTempFile("nadata-java-delta", ".ndb");
            Files.deleteIfExists(delta_store);

            try (Nadata.Database fresh = Nadata.Database.load(ex0)) {
                fresh.dump(delta_store.toString());
            }

            db.setAttribute("tom", "birthplace", "Nice");
            long docs = db.saveChanges(delta_store.toString());
            ok("nd_save_changes/delta_appended", docs >= 1);
            db.setAttribute("tom", "birthplace", "Paris");

            /* nd_compact */
            Path compact_store = Files.createTempFile("nadata-java-c", ".ndb");
            Files.deleteIfExists(compact_store);

            try (Nadata.Database fresh = Nadata.Database.load(ex0)) {
                fresh.dump(compact_store.toString());
            }

            try (Nadata.Database reopened = Nadata.Database.open(
                    delta_store.toString())) {
                long[] counts = reopened.compact(null);
                ok("nd_compact/in_place_returns_counts", counts[0] >= 0);
            }

            /* nd_set_loop_limit */
            db.setLoopLimit(100000);
            System.out.println("ok   nd_set_loop_limit/accepted");

            /* nd_execute_schema + data + by-id usage */
            db.executeSchema("class Weapon { name: Str = \"Sword\"; }");
            long excalibur = db.createInstance("Weapon", null, "exc");
            ok("nd_execute_schema/merge", excalibur >= 0);
            ok("nd_execute_data/instances",
                db.getCount("Weapon") == 1);

            /* Choice B -- by-entity-id surface */
            ok("nd_get_attr_by_id/ok",
                "Paris".equals(db.getAttrById(tom, "birthplace")));

            okStatus("nd_get_attr_by_id/unknown_id", NadataLibrary.ND_ERR_NOTFOUND,
                () -> { db.getAttrById(12345, "name"); return null; });

            okStatus("nd_get_attr_by_id/unknown_attribute",
                NadataLibrary.ND_ERR_NOTFOUND,
                () -> { db.getAttrById(tom, "no_such"); return null; });

            db.setAttrById(tom, "birthplace", "Nice");
            ok("nd_set_attr_by_id/roundtrip",
                "Nice".equals(db.getAttrById(tom, "birthplace")));
            db.setAttrById(tom, "birthplace", "Paris");

            ok("nd_call_method_by_id/ok",
                Boolean.TRUE.equals(db.callMethodById(
                    db.getSymbolId("jean"), "older", Nadata.instance(tom))));

            okStatus("nd_call_method_by_id/unknown_method",
                NadataLibrary.ND_ERR_NOTFOUND,
                () -> { db.callMethodById(tom, "no_such_method"); return null; });

            ok("nd_list_attr/ok", db.listAttr(tom).contains("birthplace"));
            ok("nd_list_methods/ok", db.listMethods(tom).contains("older"));

            ok("nd_instance_id_valid/true", db.instanceIdValid(tom));
            ok("nd_instance_id_valid/false", !db.instanceIdValid(12345));

            /* nd_get_by_nid -- session project via nid binding */
            Path session = Files.createTempDirectory("nadata-java-nid");
            write_file(session, "structs/schema.nastruct",
                "class Person {\n    name: Str;\n    age: Int32 = 0;\n}\n");
            write_file(session, "data/seed.nadata",
                "#init\n"
                + "tom = Person(nid = \"tommy\", name = \"Tom\", age = 6);\n"
                + "flush();\n");

            try (Nadata.Database nid_db = Nadata.Database.load(
                    session.toString())) {
                OptionalLong found = nid_db.getByNid("Person", "tommy");
                ok("nd_get_by_nid/ok", found.isPresent());
                ok("nd_get_by_nid/readback", found.isPresent()
                    && "Tom".equals(nid_db.getAttrById(
                        found.getAsLong(), "name")));
                ok("nd_get_by_nid/unknown_nid",
                    !nid_db.getByNid("Person", "nobody").isPresent());
            }
        }

        System.out.println(failures == 0
            ? "all cases passed" : failures + " failure(s)");
        System.exit(failures == 0 ? 0 : 1);
    }
}
