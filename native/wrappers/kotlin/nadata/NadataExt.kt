/*
 * NadataExt.kt — idiomatic Kotlin layer over the Java/JNA binding.
 *
 * Zero new native code: this file only adds Kotlin-shaped access to
 * nadata.Nadata (operators, properties, Long? instead of OptionalLong,
 * try/catch-free resource management). Extensions compile to static
 * methods calling the same Java methods the benchmark measures, so
 * performance is identical to the Java binding by construction.
 *
 *   Nadata.loadLibrary(path)
 *   Nadata.Database.load("./examples/ex0_simple").using { db ->
 *       db["tom", "birthplace"] = "Lyon"
 *       db.query("Human").apply { whereStr("h", "h.name == \"Tom\"") }.count()
 *   }
 */

package nadata.kotlin

import nadata.Nadata
import nadata.NadataLibrary

/* ------------------------------------------------------------------ */
/* Bootstrap + resource management                                     */
/* ------------------------------------------------------------------ */

/** Loads the native shim; must be called before any Database. */
fun loadLibrary(path: String) {
    Nadata.loadLibrary(path)
}

/**
 * `use {}` for AutoCloseable without depending on a specific stdlib
 * jdk7 artifact: Database and Query are AutoCloseable in Java.
 */
inline fun <T : AutoCloseable, R> T.using(block: (T) -> R): R {
    try {
        return block(this)
    } finally {
        close()
    }
}

/* ------------------------------------------------------------------ */
/* Database — symbol-keyed surface                                     */
/* ------------------------------------------------------------------ */

/** Reads an attribute: `db["tom", "birthplace"]`. */
operator fun Nadata.Database.get(symbol: String, attribute: String): Any? =
    getAttribute(symbol, attribute)

/** Writes an attribute: `db["tom", "birthplace"] = "Lyon"`. */
operator fun Nadata.Database.set(symbol: String, attribute: String,
                                 value: Any?) {
    setAttribute(symbol, attribute, value)
}

/**
 * Reads an attribute, mapping ND_ERR_NOTFOUND to null and rethrowing
 * every other status as [Nadata.Error].
 */
fun Nadata.Database.getOrNull(symbol: String, attribute: String): Any? =
    try {
        getAttribute(symbol, attribute)
    } catch (error: Nadata.Error) {
        if (error.status == NadataLibrary.ND_ERR_NOTFOUND) null
        else throw error
    }

val Nadata.Database.structNames: List<String> get() = getStructNames()
val Nadata.Database.functionNames: List<String> get() = getFunctionNames()
val Nadata.Database.symbolNames: List<String> get() = getSymbolNames()
val Nadata.Database.summary: String get() = getSummary()
val Nadata.Database.totalInstances: Long get() = getTotalInstanceCount()

fun Nadata.Database.countOf(structName: String): Long = getCount(structName)

fun Nadata.Database.symbolId(symbol: String): Long = getSymbolId(symbol)

/** Instance reference for callFunction/callMethod arguments. */
fun instance(id: Long): Nadata.Instance = Nadata.instance(id)

/* ------------------------------------------------------------------ */
/* Database — by-entity-id surface (ABI 2)                             */
/* ------------------------------------------------------------------ */

/** Reads an attribute by entity id: `db[tomId, "birthplace"]`. */
operator fun Nadata.Database.get(id: Long, attribute: String): Any? =
    getAttrById(id, attribute)

/** Writes an attribute by entity id. */
operator fun Nadata.Database.set(id: Long, attribute: String, value: Any?) {
    setAttrById(id, attribute, value)
}

/** ND_ERR_NOTFOUND -> null (optional by contract). */
fun Nadata.Database.getAttrByIdOrNull(id: Long, attribute: String): Any? =
    try {
        getAttrById(id, attribute)
    } catch (error: Nadata.Error) {
        if (error.status == NadataLibrary.ND_ERR_NOTFOUND) null
        else throw error
    }

/* ------------------------------------------------------------------ */
/* Query — optionals + chaining                                        */
/* ------------------------------------------------------------------ */

/** OptionalLong -> Kotlin null (ND_ERR_NOTFOUND is "empty"). */
fun Nadata.Query.firstIdOrNull(): Long? =
    firstId().takeIf { it.isPresent }?.asLong

/** OptionalLong -> Kotlin null (ND_ERR_NOTFOUND is "empty"). */
fun Nadata.Query.lastIdOrNull(): Long? =
    lastId().takeIf { it.isPresent }?.asLong

/** OptionalLong -> Kotlin null (ND_ERR_NOTFOUND is "empty"). */
fun Nadata.Database.getByNidOrNull(baseStruct: String, nid: String): Long? =
    getByNid(baseStruct, nid).takeIf { it.isPresent }?.asLong

/**
 * Chaining is rebind-in-place (the C ABI has no query builder), so the
 * Kotlin shape is `db.query("Human").apply {
 *     whereStr("h", "h.name == \"Tom\"")
 * }.count()`.
 */
