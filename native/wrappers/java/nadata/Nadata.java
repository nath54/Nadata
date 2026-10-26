package nadata;

import com.sun.jna.FunctionMapper;
import com.sun.jna.Library;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.NativeLibrary;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.DoubleByReference;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;

import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;

/**
 * The Nadata Java binding — a thin, Java-idiomatic layer over the C
 * ABI (docs/native.md). Mirrors the C++ reference binding:
 *
 * <pre>
 * try (Nadata.Database db = Nadata.Database.load("./my_project")) {
 *     String birthplace = db.getAttribute("tom", "birthplace");
 *     long tom = db.getSymbolId("tom");
 *     db.setAttribute("tom", "birthplace", "Lyon");
 *     long count = db.query("Human").whereStr("h", "h.name == \"Tom\"").count();
 *     OptionalLong id = db.query("Human").firstId();   // empty if none
 *     List&lt;Long&gt; ids = db.query("Human").toListIds();
 *
 *     // ABI 2: entity ids as object handles
 *     List&lt;String&gt; methods = db.listMethods(tom);
 *     db.setAttrById(tom, "name", "Tom");
 * }
 * </pre>
 *
 * Values map to natural Java types:
 *
 * <ul>
 *   <li>inputs: {@link String}, {@link Long}/{@link Integer},
 *       {@link Double}/{@link Float}, {@link Boolean}, {@link List},
 *       {@link Map} with String keys, and {@link #instance(long)} for
 *       entity ids</li>
 *   <li>outputs: {@link String}, {@link Long}, {@link Double},
 *       {@link Boolean}, {@link List}, {@link Map}; entity references
 *       come back as raw {@link Long} entity ids (use
 *       {@link #instance(long)} to pass them back in). Ids are opaque
 *       64-bit values — 0 can be a valid id, check
 *       {@link Database#instanceIdValid} instead of comparing to 0.</li>
 * </ul>
 *
 * {@link DateTime}/{@link Color} engine values are not representable
 * in ABI 2 — such calls throw {@link Nadata.Error} with status
 * ND_ERR_TYPE (keep that logic in nadata-language scripts).
 */
public final class Nadata {

    /* ------------------------------------------------------------------ */
    /* Library loading                                                     */
    /* ------------------------------------------------------------------ */

    private static final int ND_VALUE_SIZE = 32;   /* tag + union on 64-bit */

    private static volatile Lib library = null;
    private static final Object LIB_LOCK = new Object();

    /** Resolves camelCase Java methods to their nd_* C symbols. */
    private static final FunctionMapper SNAKE_CASE = new FunctionMapper() {
        @Override
        public String getFunctionName(NativeLibrary library, Method method) {
            String name = method.getName();

            if (name.startsWith("nd_")) {
                return name;   // already the exact C symbol
            }

            StringBuilder builder = new StringBuilder("nd_");

            for (char c : name.toCharArray()) {
                if (Character.isUpperCase(c)) {
                    builder.append('_').append(Character.toLowerCase(c));
                } else {
                    builder.append(c);
                }
            }

            return builder.toString();
        }
    };

    /** The mapped C entry points (all resolved lazily by name). */
    /** int (Pointer, const char*, NdValue*) — the aggregate shape. */
    interface AggregateFn {
        int query(Pointer qs, String attribute, Pointer out);
    }

    interface Lib extends Library {
        Pointer nd_runtime_info();

        void nd_free_value(Pointer value);
        void nd_free_str(Pointer str);

        int nd_load(String path, PointerByReference out_db);
        int nd_open(String storage_path, PointerByReference out_db);
        int nd_open_folder(String folder_path, PointerByReference out_db);
        void nd_close(Pointer db);

        int nd_struct_names(Pointer db, Pointer out);
        int nd_function_names(Pointer db, Pointer out);
        int nd_symbol_names(Pointer db, Pointer out);
        int nd_summary(Pointer db, PointerByReference out);
        int nd_count(Pointer db, String struct_name, LongByReference out);
        int nd_total_instance_count(Pointer db, LongByReference out);
        /* (query counting is nd_query_count below) */

        int nd_get_symbol_id(Pointer db, String symbol, LongByReference out);
        int nd_get_attribute(Pointer db, String symbol, String name,
                             Pointer out);
        int nd_set_attribute(Pointer db, String symbol, String name,
                             Pointer value);
        int nd_create_instance(Pointer db, String struct_name,
                               Pointer attributes, String symbol,
                               LongByReference out_id);
        int nd_call_function(Pointer db, String name, Pointer args,
                             Pointer kwargs, Pointer out);
        int nd_call_method(Pointer db, String symbol, String name,
                           Pointer args, Pointer kwargs, Pointer out);

        int nd_query_new(Pointer db, String struct_name,
                         PointerByReference out);
        int nd_query_where_str(Pointer qs, String var, String text);
        int nd_query_order_by(Pointer qs, String attribute, int descending);
        int nd_query_limit(Pointer qs, long n);
        int nd_query_count(Pointer qs, LongByReference out);
        int nd_query_exists(Pointer qs, IntByReference out);
        int nd_query_first_id(Pointer qs, LongByReference out);
        int nd_query_last_id(Pointer qs, LongByReference out);
        int nd_query_pluck(Pointer qs, String attribute, Pointer out);
        /* sum/min/max output a full NdValue (natural numeric type).
         * Only nd_query_average_of writes a raw double. */
        int nd_query_sum_of(Pointer qs, String attribute, Pointer out);
        int nd_query_min_of(Pointer qs, String attribute, Pointer out);
        int nd_query_max_of(Pointer qs, String attribute, Pointer out);
        int nd_query_average_of(Pointer qs, String attribute,
                                DoubleByReference out);
        int nd_query_to_list_ids(Pointer qs, Pointer out);
        void nd_query_free(Pointer qs);

        int nd_dump(Pointer db, String storage_path);
        int nd_save_changes(Pointer db, String storage_path,
                            LongByReference out_docs);
        int nd_compact(Pointer db, String output_path,
                       LongByReference out_before, LongByReference out_after);
        int nd_set_loop_limit(Pointer db, long limit);

        int nd_execute_schema(Pointer db, String schema_text);
        int nd_execute_data(Pointer db, String data_text);
        int nd_execute(Pointer db, String text);
        int nd_execute_script_file(Pointer db, String file_path,
                                   Pointer params);

        Pointer nd_last_error(Pointer db);   /* const char*, no free */

        int nd_get_attr_by_id(Pointer db, long id, String name, Pointer out);
        int nd_set_attr_by_id(Pointer db, long id, String name,
                              Pointer value);
        int nd_call_method_by_id(Pointer db, long id, String name,
                                 Pointer args, Pointer kwargs, Pointer out);
        int nd_list_attr(Pointer db, long id, Pointer out);
        int nd_list_methods(Pointer db, long id, Pointer out);
        int nd_instance_id_valid(Pointer db, long id, IntByReference out);
        int nd_get_by_nid(Pointer db, String base_struct, String nid,
                          LongByReference out);
    }

    /** Loads the shared library from an explicit path (or name). */
    public static Lib loadLibrary(String path) {
        synchronized (LIB_LOCK) {
            if (library == null) {
                Map<String, Object> options = new LinkedHashMap<>();
                options.put(Library.OPTION_FUNCTION_MAPPER, SNAKE_CASE);
                library = Native.load(path, Lib.class, options);
            }

            return library;
        }
    }

    private static Lib lib() {
        synchronized (LIB_LOCK) {
            if (library == null) {
                throw new IllegalStateException(
                    "call Nadata.loadLibrary(<path to nadata library>) "
                    + "before creating a Database");
            }

            return library;
        }
    }

    private Nadata() {
    }

    /* ------------------------------------------------------------------ */
    /* Errors                                                              */
    /* ------------------------------------------------------------------ */

    /** The ABI failure mode: an ND status plus the engine's message. */
    public static final class Error extends RuntimeException {
        private static final long serialVersionUID = 1L;

        public final int status;

        Error(int status, String message) {
            super(message);
            this.status = status;
        }
    }

    private static void require(int status, Pointer db, String call) {
        if (status == 0) {
            return;
        }

        Pointer message = lib().nd_last_error(db);
        String text = message != null ? message.getString(0) : "";

        throw new Error(status, call + ": " + text);
    }

    private static void requireNoDb(int status, String call) {
        if (status == 0) {
            return;
        }

        throw new Error(status, call);
    }

    /* ------------------------------------------------------------------ */
    /* Entity-id marker                                                    */
    /* ------------------------------------------------------------------ */

    /** An entity id read from the engine (or built for input). */
    public static final class Instance {
        public final long id;

        private Instance(long id) {
            this.id = id;
        }

        @Override
        public String toString() {
            return "Instance(" + id + ")";
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof Instance
                && ((Instance) other).id == id;
        }

        @Override
        public int hashCode() {
            return Long.hashCode(id);
        }
    }

    /** Marks a long as an entity id (ND_INSTANCE) for input values. */
    public static Instance instance(long id) {
        return new Instance(id);
    }

    /* ------------------------------------------------------------------ */
    /* Value packing (inputs) — 32-byte NdValue blocks                     */
    /* ------------------------------------------------------------------ */

    private static Memory newBlock() {
        return new Memory(ND_VALUE_SIZE);
    }

    /** A NUL-terminated UTF-8 copy of text in fresh native memory. */
    private static Memory nativeString(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        Memory block = new Memory(bytes.length + 1);

        block.write(0, bytes, 0, bytes.length);
        block.setByte(bytes.length, (byte) 0);

        return block;
    }

    private static void setTag(Pointer block, int tag) {
        block.setInt(0, tag);
    }

    /** Packs a value into a fresh 32-byte block (pinned via keep). */
    private static Memory pack(Object value, List<Object> keep) {
        Memory block = newBlock();
        packInto(value, block, 0, keep);
        return block;
    }

    /** Packs a value at offset 0 of an existing block. */
    private static void packInto(Object value, Memory block,
                                 List<Object> keep) {
        packInto(value, block, 0, keep);
    }

    /** Writes any input value into block[offset .. offset+32). */
    private static void packInto(Object value, Memory block, long offset,
                                 List<Object> keep) {
        if (value instanceof Instance) {
            setTag(block, offset, NadataLibrary.ND_INSTANCE);
            block.setLong(offset + 8, ((Instance) value).id);

        } else if (value instanceof String) {
            Memory string = nativeString((String) value);
            keep.add(string);
            setTag(block, offset, NadataLibrary.ND_STR);
            block.setPointer(offset + 8, string);

        } else if (value instanceof Boolean) {
            setTag(block, offset, NadataLibrary.ND_BOOL);
            block.setInt(offset + 8, (Boolean) value ? 1 : 0);

        } else if (value instanceof Double) {
            setTag(block, offset, NadataLibrary.ND_FLOAT64);
            block.setDouble(offset + 8, (Double) value);

        } else if (value instanceof Float) {
            packInto((double) (Float) value, block, offset, keep);

        } else if (value instanceof Long) {
            setTag(block, offset, NadataLibrary.ND_INT64);
            block.setLong(offset + 8, (Long) value);

        } else if (value instanceof Integer) {
            packInto(((Integer) value).longValue(), block, offset, keep);

        } else if (value instanceof List) {
            List<?> items = (List<?>) value;
            Memory block_items = new Memory(
                ND_VALUE_SIZE * Math.max(items.size(), 1));

            for (int i = 0; i < items.size(); i++) {
                packInto(items.get(i), block_items,
                         (long) i * ND_VALUE_SIZE, keep);
            }

            keep.add(block_items);
            setTag(block, offset, NadataLibrary.ND_ARRAY);
            block.setPointer(offset + 8, block_items);
            block.setLong(offset + 16, items.size());

        } else if (value instanceof Map) {
            Map<?, ?> map = (Map<?, ?>) value;
            int count = map.size();
            Memory keys = new Memory(8L * Math.max(count, 1));
            Memory values = new Memory(
                ND_VALUE_SIZE * Math.max(count, 1));

            int i = 0;

            for (Map.Entry<?, ?> entry : map.entrySet()) {
                if (!(entry.getKey() instanceof String)) {
                    throw new IllegalArgumentException(
                        "dict keys must be String, got "
                        + entry.getKey().getClass());
                }

                Memory string = nativeString((String) entry.getKey());
                keep.add(string);
                keys.setPointer((long) i * 8, string);
                packInto(entry.getValue(), values,
                         (long) i * ND_VALUE_SIZE, keep);
                i++;
            }

            keep.add(keys);
            keep.add(values);
            setTag(block, offset, NadataLibrary.ND_DICT);
            block.setPointer(offset + 8, keys);
            block.setPointer(offset + 16, values);
            block.setLong(offset + 24, count);

        } else {
            throw new IllegalArgumentException(
                "unsupported value type: " + value.getClass()
                + " (use Nadata.instance(id) for entity ids)");
        }
    }

    private static void setTag(Pointer block, long offset, int tag) {
        block.setInt(offset, tag);
    }

    private static final Object[] NO_ARGS = new Object[0];

    /** Packs optional args/kwargs; returns {args, kwargs} blocks. */
    private static Memory[] packCallArgs(Object[] args,
                                         Map<String, Object> kwargs,
                                         List<Object> keep) {
        Memory args_block = null;

        if (args != null && args.length > 0) {
            args_block = newBlock();
            packInto(java.util.Arrays.asList(args), args_block, 0, keep);
        }

        Memory kwargs_block = null;

        if (kwargs != null && !kwargs.isEmpty()) {
            kwargs_block = newBlock();
            packInto(kwargs, kwargs_block, 0, keep);
        }

        return new Memory[] {args_block, kwargs_block};
    }

    /* ------------------------------------------------------------------ */
    /* Value reading (outputs)                                             */
    /* ------------------------------------------------------------------ */

    /** Reads an output block into Java objects, then frees it. */
    private static Object readOutput(Lib lib, Memory block) {
        Object value = readValue(block, 0);
        lib.nd_free_value(block);
        return value;
    }

    private static Object readValue(Pointer block, long offset) {
        int tag = block.getInt(offset);

        switch (tag) {
            case NadataLibrary.ND_NULL:
                return null;

            case NadataLibrary.ND_BOOL:
                return block.getInt(offset + 8) != 0;

            case NadataLibrary.ND_INT64:
            case NadataLibrary.ND_INSTANCE:
                return block.getLong(offset + 8);

            case NadataLibrary.ND_FLOAT64:
                return block.getDouble(offset + 8);

            case NadataLibrary.ND_STR:
                return block.getPointer(offset + 8).getString(0);

            case NadataLibrary.ND_ARRAY: {
                Pointer items = block.getPointer(offset + 8);
                long count = block.getLong(offset + 16);
                List<Object> list = new ArrayList<>((int) count);

                for (long i = 0; i < count; i++) {
                    list.add(readValue(items, i * ND_VALUE_SIZE));
                }

                return list;
            }

            case NadataLibrary.ND_DICT: {
                Pointer keys = block.getPointer(offset + 8);
                Pointer values = block.getPointer(offset + 16);
                long count = block.getLong(offset + 24);
                Map<String, Object> map = new LinkedHashMap<>((int) count);

                for (long i = 0; i < count; i++) {
                    String key = keys.getPointer(i * 8).getString(0);
                    map.put(key, readValue(values, i * ND_VALUE_SIZE));
                }

                return map;
            }

            default:
                throw new Error(NadataLibrary.ND_ERR_STATE,
                    "unknown NdValue tag " + tag + " in output");
        }
    }

    /** Reads an ND_ARRAY output as a list of entity ids, then frees. */
    private static List<Long> readIds(Lib lib, Memory block) {
        Object value = readOutput(lib, block);

        if (!(value instanceof List)) {
            throw new Error(NadataLibrary.ND_ERR_STATE,
                "expected an id array, got " + value);
        }

        List<Long> ids = new ArrayList<>();

        for (Object entry : (List<?>) value) {
            if (entry instanceof Long) {
                ids.add((Long) entry);
            } else {
                throw new Error(NadataLibrary.ND_ERR_STATE,
                    "expected an id array, got element " + entry);
            }
        }

        return ids;
    }

    /* ------------------------------------------------------------------ */
    /* Database                                                            */
    /* ------------------------------------------------------------------ */

    /** An open database (maps to nadata.api.Nadata). */
    public static final class Database implements AutoCloseable {
        private final Pointer db;

        private Database(Pointer db) {
            this.db = db;
        }

        /** Full project load (schema folders, files, or .ndb). */
        public static Database load(String path) {
            PointerByReference out = new PointerByReference();
            requireNoDb(lib().nd_load(path, out), "nd_load");
            return new Database(out.getValue());
        }

        /** Opens an existing .ndb dump. */
        public static Database open(String storage_path) {
            PointerByReference out = new PointerByReference();
            requireNoDb(lib().nd_open(storage_path, out), "nd_open");
            return new Database(out.getValue());
        }

        /** Opens a session folder. */
        public static Database openFolder(String folder_path) {
            PointerByReference out = new PointerByReference();
            requireNoDb(lib().nd_open_folder(folder_path, out), "nd_open_folder");
            return new Database(out.getValue());
        }

        /** Runtime info JSON (abi, cpython version, build). */
        public static String runtimeInfo() {
            Pointer info = lib().nd_runtime_info();

            if (info == null) {
                throw new Error(NadataLibrary.ND_ERR_RUNTIME,
                    "nd_runtime_info failed");
            }

            String json = info.getString(0);
            lib().nd_free_str(info);
            return json;
        }

        public List<String> getStructNames() {
            Memory out = newBlock();
            require(lib().nd_struct_names(db, out), db, "nd_struct_names");
            return readStringList(out);
        }

        public List<String> getFunctionNames() {
            Memory out = newBlock();
            require(lib().nd_function_names(db, out), db, "nd_function_names");
            return readStringList(out);
        }

        public List<String> getSymbolNames() {
            Memory out = newBlock();
            require(lib().nd_symbol_names(db, out), db, "nd_symbol_names");
            return readStringList(out);
        }

        public String getSummary() {
            PointerByReference out = new PointerByReference();
            require(lib().nd_summary(db, out), db, "nd_summary");

            Pointer text = out.getValue();
            String summary = text.getString(0);
            lib().nd_free_str(text);
            return summary;
        }

        public long getCount(String struct_name) {
            LongByReference out = new LongByReference();
            require(lib().nd_count(db, struct_name, out), db, "nd_count");
            return out.getValue();
        }

        public long getTotalInstanceCount() {
            LongByReference out = new LongByReference();
            require(lib().nd_total_instance_count(db, out), db,
                "nd_total_instance_count");
            return out.getValue();
        }

        public long getSymbolId(String symbol) {
            LongByReference out = new LongByReference();
            require(lib().nd_get_symbol_id(db, symbol, out), db,
                "nd_get_symbol_id");
            return out.getValue();
        }

        public Object getAttribute(String symbol, String name) {
            Memory out = newBlock();
            require(lib().nd_get_attribute(db, symbol, name, out), db,
                "nd_get_attribute");
            return readOutput(lib(), out);
        }

        public void setAttribute(String symbol, String name, Object value) {
            List<Object> keep = new ArrayList<>();
            Memory packed = pack(value, keep);

            require(lib().nd_set_attribute(db, symbol, name, packed), db,
                "nd_set_attribute");
        }

        /** Creates an instance; returns its entity id. */
        public long createInstance(String struct_name,
                                   Map<String, Object> attributes,
                                   String symbol) {
            List<Object> keep = new ArrayList<>();
            Memory packed_attributes =
                attributes != null && !attributes.isEmpty()
                    ? pack(attributes, keep) : null;

            LongByReference out = new LongByReference();
            require(lib().nd_create_instance(db, struct_name,
                packed_attributes, symbol, out), db, "nd_create_instance");
            return out.getValue();
        }

        /** Calls an engine function; args as plain Java values. */
        public Object callFunction(String name, Object... args) {
            List<Object> keep = new ArrayList<>();
            Memory[] packed = packCallArgs(args, null, keep);

            Memory out = newBlock();
            require(lib().nd_call_function(db, name,
                packed[0], packed[1], out), db, "nd_call_function");
            return readOutput(lib(), out);
        }

        /** Calls an engine function with kwargs. */
        public Object callFunction(String name, Object[] args,
                                   Map<String, Object> kwargs) {
            List<Object> keep = new ArrayList<>();
            Memory[] packed = packCallArgs(args, kwargs, keep);

            Memory out = newBlock();
            require(lib().nd_call_function(db, name,
                packed[0], packed[1], out), db, "nd_call_function");
            return readOutput(lib(), out);
        }

        public Object callMethod(String symbol, String name, Object... args) {
            List<Object> keep = new ArrayList<>();
            Memory[] packed = packCallArgs(args, null, keep);

            Memory out = newBlock();
            require(lib().nd_call_method(db, symbol, name,
                packed[0], packed[1], out), db, "nd_call_method");
            return readOutput(lib(), out);
        }

        /* ------------------------- persistence ------------------------ */

        public void dump(String storage_path) {
            require(lib().nd_dump(db, storage_path), db, "nd_dump");
        }

        /** Appends a delta; returns the documents written. */
        public long saveChanges(String storage_path) {
            LongByReference out = new LongByReference();
            require(lib().nd_save_changes(db, storage_path, out), db,
                "nd_save_changes");
            return out.getValue();
        }

        /** Compacts in place (or to output_path); returns {before, after}. */
        public long[] compact(String output_path) {
            LongByReference before = new LongByReference();
            LongByReference after = new LongByReference();
            require(lib().nd_compact(db, output_path, before, after), db,
                "nd_compact");
            return new long[] {before.getValue(), after.getValue()};
        }

        public void setLoopLimit(long limit) {
            require(lib().nd_set_loop_limit(db, limit), db,
                "nd_set_loop_limit");
        }

        public void executeSchema(String schema_text) {
            require(lib().nd_execute_schema(db, schema_text), db,
                "nd_execute_schema");
        }

        public void executeData(String data_text) {
            require(lib().nd_execute_data(db, data_text), db,
                "nd_execute_data");
        }

        public void execute(String text) {
            require(lib().nd_execute(db, text), db, "nd_execute");
        }

        public void executeScriptFile(String file_path,
                                      Map<String, Object> params) {
            List<Object> keep = new ArrayList<>();
            Memory packed = params != null && !params.isEmpty()
                ? pack(params, keep) : null;

            require(lib().nd_execute_script_file(db, file_path, packed),
                db, "nd_execute_script_file");
        }

        /* ---------------------- by-entity-id (ABI 2) ------------------ */

        public Object getAttrById(long id, String name) {
            Memory out = newBlock();
            require(lib().nd_get_attr_by_id(db, id, name, out), db,
                "nd_get_attr_by_id");
            return readOutput(lib(), out);
        }

        public void setAttrById(long id, String name, Object value) {
            List<Object> keep = new ArrayList<>();
            Memory packed = pack(value, keep);

            require(lib().nd_set_attr_by_id(db, id, name, packed), db,
                "nd_set_attr_by_id");
        }

        public Object callMethodById(long id, String name, Object... args) {
            List<Object> keep = new ArrayList<>();
            Memory[] packed = packCallArgs(args, null, keep);

            Memory out = newBlock();
            require(lib().nd_call_method_by_id(db, id, name,
                packed[0], packed[1], out), db, "nd_call_method_by_id");
            return readOutput(lib(), out);
        }

        public List<String> listAttr(long id) {
            Memory out = newBlock();
            require(lib().nd_list_attr(db, id, out), db, "nd_list_attr");
            return readStringList(out);
        }

        public List<String> listMethods(long id) {
            Memory out = newBlock();
            require(lib().nd_list_methods(db, id, out), db, "nd_list_methods");
            return readStringList(out);
        }

        public boolean instanceIdValid(long id) {
            IntByReference out = new IntByReference();
            require(lib().nd_instance_id_valid(db, id, out), db,
                "nd_instance_id_valid");
            return out.getValue() != 0;
        }

        /** Resolves a struct nid; empty when unknown. */
        public OptionalLong getByNid(String base_struct, String nid) {
            LongByReference out = new LongByReference();
            int status = lib().nd_get_by_nid(db, base_struct, nid, out);

            if (status == NadataLibrary.ND_ERR_NOTFOUND) {
                return OptionalLong.empty();
            }

            require(status, db, "nd_get_by_nid");
            return OptionalLong.of(out.getValue());
        }

        /* ----------------------------- misc --------------------------- */

        /** Starts a query (rebind-in-place semantics on the handle). */
        public Query query(String struct_name) {
            PointerByReference out = new PointerByReference();
            require(lib().nd_query_new(db, struct_name, out), db,
                "nd_query_new");
            return new Query(out.getValue());
        }

        @Override
        public void close() {
            lib().nd_close(db);
        }

        private Pointer db() {
            return db;
        }
    }

    /* ------------------------------------------------------------------ */
    /* Query — rebind-in-place (filters return this)                       */
    /* ------------------------------------------------------------------ */

    /** A query handle; filters rebind it in place, chain freely. */
    public static final class Query implements AutoCloseable {
        private final Pointer qs;

        private Query(Pointer qs) {
            this.qs = qs;
        }

        /** nadata-language predicate: whereStr("h", "h.age >= 18"). */
        public Query whereStr(String var, String text) {
            require(lib().nd_query_where_str(qs, var, text), qs,
                "nd_query_where_str");
            return this;
        }

        public Query orderBy(String attribute, boolean descending) {
            require(lib().nd_query_order_by(qs, attribute,
                descending ? 1 : 0), qs, "nd_query_order_by");
            return this;
        }

        /** n < 1 throws Error(ND_ERR_INVALID_ARG). */
        public Query limit(long n) {
            require(lib().nd_query_limit(qs, n), qs, "nd_query_limit");
            return this;
        }

        public long count() {
            LongByReference out = new LongByReference();
            require(lib().nd_query_count(qs, out), qs, "nd_query_count");
            return out.getValue();
        }

        public boolean exists() {
            IntByReference out = new IntByReference();
            require(lib().nd_query_exists(qs, out), qs, "nd_query_exists");
            return out.getValue() != 0;
        }

        /** Empty when the result set is empty (ids are 0-based). */
        public OptionalLong firstId() {
            LongByReference out = new LongByReference();
            int status = lib().nd_query_first_id(qs, out);

            if (status == NadataLibrary.ND_ERR_NOTFOUND) {
                return OptionalLong.empty();
            }

            require(status, qs, "nd_query_first_id");
            return OptionalLong.of(out.getValue());
        }

        public OptionalLong lastId() {
            LongByReference out = new LongByReference();
            int status = lib().nd_query_last_id(qs, out);

            if (status == NadataLibrary.ND_ERR_NOTFOUND) {
                return OptionalLong.empty();
            }

            require(status, qs, "nd_query_last_id");
            return OptionalLong.of(out.getValue());
        }

        /** Raw attribute payloads of every row, in result order. */
        public List<Object> pluck(String attribute) {
            Memory out = newBlock();
            require(lib().nd_query_pluck(qs, attribute, out), qs,
                "nd_query_pluck");
            return castList(readOutput(lib(), out));
        }

        /** sum/min/max write a NdValue (natural numeric type). */
        private double numericAgg(String symbol, String attribute) {
            Memory out = newBlock();
            AggregateFn fn;

            switch (symbol) {
                case "nd_query_sum_of":
                    fn = lib()::nd_query_sum_of;
                    break;
                case "nd_query_min_of":
                    fn = lib()::nd_query_min_of;
                    break;
                default:
                    fn = lib()::nd_query_max_of;
                    break;
            }

            require(fn.query(qs, attribute, out), qs, symbol);

            Object value = readOutput(lib(), out);

            if (value instanceof Long) {
                return (double) (long) (Long) value;
            }

            if (value instanceof Double) {
                return (Double) value;
            }

            throw new Error(NadataLibrary.ND_ERR_TYPE, symbol
                + ": non-numeric result");
        }

        public double sumOf(String attribute) {
            return numericAgg("nd_query_sum_of", attribute);
        }

        public double minOf(String attribute) {
            return numericAgg("nd_query_min_of", attribute);
        }

        public double maxOf(String attribute) {
            return numericAgg("nd_query_max_of", attribute);
        }

        public double averageOf(String attribute) {
            DoubleByReference out = new DoubleByReference();
            require(lib().nd_query_average_of(qs, attribute, out), qs,
                "nd_query_average_of");
            return out.getValue();
        }

        public List<Long> toListIds() {
            Memory out = newBlock();
            require(lib().nd_query_to_list_ids(qs, out), qs,
                "nd_query_to_list_ids");
            return readIds(lib(), out);
        }

        @Override
        public void close() {
            lib().nd_query_free(qs);
        }
    }

    /* ------------------------------------------------------------------ */
    /* Internal helpers                                                    */
    /* ------------------------------------------------------------------ */

    private static Memory outBlock() {
        return newBlock();
    }

    @SuppressWarnings("unchecked")
    private static List<Object> castList(Object value) {
        if (!(value instanceof List)) {
            throw new Error(NadataLibrary.ND_ERR_STATE,
                "expected a list, got " + value);
        }

        return (List<Object>) value;
    }

    /** Reads an ND_ARRAY-of-strings output block, then frees it. */
    private static List<String> readStringList(Memory block) {
        Object value = readOutput(lib(), block);

        if (!(value instanceof List)) {
            throw new Error(NadataLibrary.ND_ERR_STATE,
                "expected a string array, got " + value);
        }

        List<String> names = new ArrayList<>();

        for (Object entry : (List<?>) value) {
            if (entry instanceof String) {
                names.add((String) entry);
            } else {
                throw new Error(NadataLibrary.ND_ERR_STATE,
                    "expected a string array, got element " + entry);
            }
        }

        return names;
    }

}
