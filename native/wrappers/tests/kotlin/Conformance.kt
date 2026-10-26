/*
 * Conformance.kt — case-id replay for the Kotlin wrapper.
 *
 * Same case ids and expectations as native/wrappers/tests/java and
 * test_conformance.cpp (canonical registry: tests/test_abi_conformance.py).
 * Exercises the idiomatic layer from nadata.kotlin.NadataExt (operators,
 * properties, Long? optionals) rather than the raw Java API where a
 * case allows it.
 *
 * Usage (from the repository root): see run.py in this directory.
 */

package nadata.kotlin

import nadata.Nadata
import nadata.NadataLibrary
import java.nio.file.Files
import java.nio.file.Path

var failures = 0

fun ok(case: String, condition: Boolean) {
    println(if (condition) "ok   $case" else "FAIL $case")

    if (!condition) {
        failures++
    }
}

inline fun okStatus(case: String, expected: Int, block: () -> Unit) {
    try {
        block()
        println("FAIL $case (no exception)")
        failures++
    } catch (error: Nadata.Error) {
        if (error.status == expected) {
            println("ok   $case")
        } else {
            println("FAIL $case status ${error.status} != $expected: "
                + error.message)
            failures++
        }
    } catch (error: Exception) {
        println("FAIL $case unexpected: $error")
        failures++
    }
}

fun writeFile(dir: Path, rel: String, text: String) {
    val target = dir.resolve(rel)
    Files.createDirectories(target.parent)
    Files.write(target, text.toByteArray(Charsets.UTF_8))
}

fun main(args: Array<String>) {
    System.setOut(java.io.PrintStream(System.out, true, "UTF-8"))

    val libPath = System.getProperty("nadata.library") ?: run {
        val os = System.getProperty("os.name", "").lowercase()
        val dir = if (os.contains("win")) "win32"
                  else if (os.contains("mac")) "darwin" else "linux"
        val file = if (os.contains("win")) "nadata.dll"
                   else if (os.contains("mac")) "libnadata.dylib"
                   else "libnadata.so"
        Path.of("native", "build", dir, file).toString()
    }

    loadLibrary(libPath)

    val ex0 = args.getOrElse(0) { "./examples/ex0_simple" }

    /* nd_runtime_info/valid_json */
    val info = Nadata.Database.runtimeInfo()
    ok("nd_runtime_info/valid_json",
        info.contains("\"abi\":2") && info.contains("\"cpython\":"))

    /* nd_load/unsupported_path */
    okStatus("nd_load/unsupported_path", NadataLibrary.ND_ERR_VALUE) {
        Nadata.Database.load("./nadata/__init__.py")
    }

    Nadata.Database.load(ex0).using { db ->
        /* nd_load/project_folder */
        println("ok   nd_load/project_folder")

        /* nd_struct_names/ex0 */
        ok("nd_struct_names/ex0", "Human" in db.structNames)

        /* nd_function_names/ex0 */
        ok("nd_function_names/ex0", "is_adult" in db.functionNames)

        /* nd_symbol_names/ex0 */
        ok("nd_symbol_names/ex0", "tom" in db.symbolNames && "jean" in db.symbolNames)

        /* nd_summary/nonempty_string */
        ok("nd_summary/nonempty_string", db.summary.isNotEmpty())

        /* nd_count/polymorphic + nd_total_instance_count */
        ok("nd_count/polymorphic", db.countOf("Entity") == 4L)
        ok("nd_total_instance_count/ex0", db.totalInstances == 6L)

        /* nd_count/unknown_struct */
        okStatus("nd_count/unknown_struct", NadataLibrary.ND_ERR_NOTFOUND) {
            db.countOf("Nope")
        }

        /* nd_get_symbol_id */
        val tom = db.symbolId("tom")
        ok("nd_get_symbol_id/ok", tom >= 0)

        okStatus("nd_get_symbol_id/unknown_symbol",
            NadataLibrary.ND_ERR_NOTFOUND) {
            db.symbolId("nobody")
        }

        /* nd_get_attribute + operator get */
        ok("nd_get_attribute/plain_scalar", db["tom", "birthplace"] == "Paris")

        okStatus("nd_get_attribute/unknown_symbol",
            NadataLibrary.ND_ERR_NOTFOUND) {
            db["nobody", "name"]
        }

        okStatus("nd_get_attribute/unknown_attribute",
            NadataLibrary.ND_ERR_NOTFOUND) {
            db["tom", "no_such"]
        }

        okStatus("nd_get_attribute/home_type_payload",
            NadataLibrary.ND_ERR_TYPE) {
            db["tom", "birthday"]
        }

        /* getOrNull: NOTFOUND -> null, others rethrow */
        ok("getOrNull/notfound_is_null", db.getOrNull("tom", "no_such") == null)
        ok("getOrNull/unknown_symbol_is_null",
            db.getOrNull("nobody", "name") == null)
        okStatus("getOrNull/other_status_rethrown", NadataLibrary.ND_ERR_TYPE) {
            db.getOrNull("tom", "birthday")
        }

        /* nd_set_attribute + operator set */
        db["tom", "birthplace"] = "Lyon"
        ok("nd_set_attribute/roundtrip", db["tom", "birthplace"] == "Lyon")
        db["tom", "birthplace"] = "Paris"

        okStatus("nd_set_attribute/type_mismatch", NadataLibrary.ND_ERR_TYPE) {
            db["tom", "birthplace"] = 12345L
        }

        okStatus("nd_set_attribute/unknown_symbol",
            NadataLibrary.ND_ERR_NOTFOUND) {
            db["nobody", "name"] = "x"
        }

        /* nd_create_instance */
        val shadow = db.createInstance("Cat", mapOf("name" to "Shadow"),
            "shadow")
        ok("nd_create_instance/with_attributes_and_symbol", shadow >= 0)
        ok("nd_create_instance/readback", db["shadow", "name"] == "Shadow")

        okStatus("nd_create_instance/unknown_struct",
            NadataLibrary.ND_ERR_NOTFOUND) {
            db.createInstance("Nope", null, null)
        }

        /* nd_call_function */
        ok("nd_call_function/positional_args",
            db.callFunction("is_adult", instance(tom)) == true)

        okStatus("nd_call_function/unknown_function",
            NadataLibrary.ND_ERR_NOTFOUND) {
            db.callFunction("no_such_fn")
        }

        okStatus("nd_call_function/argument_type_mismatch",
            NadataLibrary.ND_ERR_TYPE) {
            db.callFunction("is_adult", "tom")
        }

        okStatus("nd_call_function/home_type_result",
            NadataLibrary.ND_ERR_TYPE) {
            db.callFunction("calculate_age_difference", instance(tom),
                instance(tom))
        }

        /* nd_call_method */
        ok("nd_call_method/ok",
            db.callMethod("jean", "older", instance(tom)) == true)

        okStatus("nd_call_method/unknown_method",
            NadataLibrary.ND_ERR_NOTFOUND) {
            db.callMethod("tom", "no_such_method")
        }

        okStatus("nd_call_method/unknown_symbol",
            NadataLibrary.ND_ERR_NOTFOUND) {
            db.callMethod("nobody", "older")
        }

        /* nd_query_* */
        okStatus("nd_query_new/unknown_struct", NadataLibrary.ND_ERR_NOTFOUND) {
            db.query("Nope")
        }

        ok("nd_query_where_str/engine_evaluated", db.query("Human")
            .apply { whereStr("h", "h.name == \"Tom\"") }.count() == 1L)

        okStatus("nd_query_where_str/syntax_error",
            NadataLibrary.ND_ERR_SYNTAX) {
            db.query("Human").apply { whereStr("h", "h ====") }
        }

        okStatus("nd_query_where_str/type_error", NadataLibrary.ND_ERR_TYPE) {
            db.query("Human").apply { whereStr("h", "h.name > 5") }
        }

        db.query("Human").apply {
            whereStr("h", "h.name == \"Nobody\"")
        }.using { empty ->
            ok("nd_query_exists/empty", !empty.exists())
            ok("nd_query_count/empty_value", empty.count() == 0L)
            ok("nd_query_first_id/empty_set_is_notfound",
                empty.firstIdOrNull() == null)
        }

        ok("nd_query_exists/true_false", db.query("Human").exists())

        /* nd_query_order_by — limit window + direction */
        db.query("Human").apply {
            orderBy("name", false)
            limit(2)
        }.using { window ->
            ok("nd_query_order_by/ascending_and_descending",
                window.toListIds().size == 2)
        }

        val ascending = db.query("Human")
            .apply { orderBy("name", false) }.toListIds()
        val descending = db.query("Human")
            .apply { orderBy("name", true) }.toListIds()
        ok("nd_query_order_by/direction",
            ascending.size == 3 && descending.size == 3
                && ascending.first() == descending.last())

        okStatus("nd_query_order_by/unknown_attribute",
            NadataLibrary.ND_ERR_NOTFOUND) {
            db.query("Human").apply { orderBy("no_such", false) }
        }

        okStatus("nd_query_limit/invalid", NadataLibrary.ND_ERR_INVALID_ARG) {
            db.query("Human").apply { limit(0) }
        }

        /* nd_query_pluck */
        val plucked = db.query("Human")
            .apply { whereStr("h", "h.name == \"Tom\"") }
            .pluck("birthplace")
        ok("nd_query_pluck/order_and_payloads",
            plucked.size == 1 && plucked[0] == "Paris")

        okStatus("nd_query_pluck/unknown_attribute",
            NadataLibrary.ND_ERR_NOTFOUND) {
            db.query("Human").pluck("no_such")
        }

        /* nd_query_aggregates — numeric fixture (numeric_project:
         * 3 Boxes, weights 1.5/2.5/3.5, counts 2/3/5). */
        val numeric = Files.createTempDirectory("nadata-kt-num")
        writeFile(numeric, "structs/schema.nastruct",
            "class Box {\n    label: Str = \"\";\n"
                + "    weight: Float64;\n    count: Int32;\n}\n")
        writeFile(numeric, "data/seed.nadata",
            "#init\n"
                + "b1 = Box(nid = \"b1\", weight = 1.5, count = 2);\n"
                + "b2 = Box(nid = \"b2\", weight = 2.5, count = 3);\n"
                + "b3 = Box(nid = \"b3\", weight = 3.5, count = 5);\n"
                + "flush();\n")

        Nadata.Database.load(numeric.toString()).using { boxes ->
            boxes.query("Box").using { all ->
                ok("nd_query_aggregates/sum",
                    kotlin.math.abs(all.sumOf("weight") - 7.5) < 1e-6)
                ok("nd_query_aggregates/min",
                    kotlin.math.abs(all.minOf("weight") - 1.5) < 1e-6)
                ok("nd_query_aggregates/max",
                    kotlin.math.abs(all.maxOf("weight") - 3.5) < 1e-6)
                ok("nd_query_aggregates/average",
                    kotlin.math.abs(all.averageOf("weight") - 2.5) < 1e-6)
            }

            okStatus("nd_query_aggregates/non_numeric_attribute",
                NadataLibrary.ND_ERR_TYPE) {
                boxes.query("Box").sumOf("label")
            }
        }

        /* nd_query_to_list_ids */
        val ids = db.query("Human").toListIds()
        ok("nd_query_to_list_ids/ordered_and_stable",
            ids.size == 3 && ids == db.query("Human").toListIds())

        /* nd_dump/nd_open roundtrip */
        val store = Files.createTempFile("nadata-kt", ".ndb")
        Files.deleteIfExists(store)

        Nadata.Database.load(ex0).using { fresh ->
            fresh.dump(store.toString())
            println("ok   nd_dump/roundtrip")

            okStatus("nd_dump/file_must_not_exist",
                NadataLibrary.ND_ERR_VALUE) {
                fresh.dump(store.toString())
            }

            Nadata.Database.open(store.toString()).using { reopened ->
                ok("nd_open/roundtrip", reopened.countOf("Human") == 3L)
            }

            okStatus("nd_open/missing_file", NadataLibrary.ND_ERR_VALUE) {
                Nadata.Database.open(store.toString() + ".missing")
            }
        }

        /* nd_save_changes */
        okStatus("nd_save_changes/missing_file", NadataLibrary.ND_ERR_VALUE) {
            db.saveChanges(store.toString() + ".missing")
        }

        val deltaStore = Files.createTempFile("nadata-kt-delta", ".ndb")
        Files.deleteIfExists(deltaStore)

        Nadata.Database.load(ex0).using { fresh ->
            fresh.dump(deltaStore.toString())
        }

        db["tom", "birthplace"] = "Nice"
        val docs = db.saveChanges(deltaStore.toString())
        ok("nd_save_changes/delta_appended", docs >= 1)
        db["tom", "birthplace"] = "Paris"

        /* nd_compact */
        val compactStore = Files.createTempFile("nadata-kt-c", ".ndb")
        Files.deleteIfExists(compactStore)

        Nadata.Database.load(ex0).using { fresh ->
            fresh.dump(compactStore.toString())
        }

        Nadata.Database.open(deltaStore.toString()).using { reopened ->
            val counts = reopened.compact(null)
            ok("nd_compact/in_place_returns_counts", counts[0] >= 0)
        }

        /* nd_set_loop_limit */
        db.setLoopLimit(100000)
        println("ok   nd_set_loop_limit/accepted")

        /* nd_execute_schema + data + by-id usage */
        db.executeSchema("class Weapon { name: Str = \"Sword\"; }")
        val excalibur = db.createInstance("Weapon", null, "exc")
        ok("nd_execute_schema/merge", excalibur >= 0)
        ok("nd_execute_data/instances", db.countOf("Weapon") == 1L)

        /* Choice B — by-entity-id surface */
        ok("nd_get_attr_by_id/ok", db.getAttrById(tom, "birthplace") == "Paris")

        okStatus("nd_get_attr_by_id/unknown_id", NadataLibrary.ND_ERR_NOTFOUND) {
            db.getAttrById(12345, "name")
        }

        okStatus("nd_get_attr_by_id/unknown_attribute",
            NadataLibrary.ND_ERR_NOTFOUND) {
            db.getAttrById(tom, "no_such")
        }

        db.setAttrById(tom, "birthplace", "Nice")
        ok("nd_set_attr_by_id/roundtrip",
            db.getAttrById(tom, "birthplace") == "Nice")
        db.setAttrById(tom, "birthplace", "Paris")

        ok("nd_call_method_by_id/ok",
            db.callMethodById(db.symbolId("jean"), "older", instance(tom))
                == true)

        okStatus("nd_call_method_by_id/unknown_method",
            NadataLibrary.ND_ERR_NOTFOUND) {
            db.callMethodById(tom, "no_such_method")
        }

        ok("nd_list_attr/ok", "birthplace" in db.listAttr(tom))
        ok("nd_list_methods/ok", "older" in db.listMethods(tom))

        ok("nd_instance_id_valid/true", db.instanceIdValid(tom))
        ok("nd_instance_id_valid/false", !db.instanceIdValid(12345))

        /* getAttrByIdOrNull: NOTFOUND -> null */
        ok("getAttrByIdOrNull/notfound_is_null",
            db.getAttrByIdOrNull(12345, "name") == null)

        /* nd_get_by_nid — session project via nid binding */
        val session = Files.createTempDirectory("nadata-kt-nid")
        writeFile(session, "structs/schema.nastruct",
            "class Person {\n    name: Str;\n    age: Int32 = 0;\n}\n")
        writeFile(session, "data/seed.nadata",
            "#init\n"
                + "tom = Person(nid = \"tommy\", name = \"Tom\", age = 6);\n"
                + "flush();\n")

        Nadata.Database.load(session.toString()).using { nidDb ->
            val found = nidDb.getByNidOrNull("Person", "tommy")
            ok("nd_get_by_nid/ok", found != null)
            ok("nd_get_by_nid/readback",
                found != null && nidDb[found, "name"] == "Tom")
            ok("nd_get_by_nid/unknown_nid",
                nidDb.getByNidOrNull("Person", "nobody") == null)
        }
    }

    println(if (failures == 0) "all cases passed"
            else "$failures failure(s)")

    kotlin.system.exitProcess(if (failures == 0) 0 else 1)
}
