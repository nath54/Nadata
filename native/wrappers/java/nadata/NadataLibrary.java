package nadata;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import com.sun.jna.Union;

/**
 * Low-level JNA mapping of the Nadata C ABI (native/nadata.h, ABI 2).
 *
 * Only the surface that is safe from a GC'd host is exposed:
 * nd_query_where_cb is intentionally NOT mapped here — the callback
 * runs with the engine's GIL attached and must not re-enter the same
 * db handle, which a GC'd callback cannot guarantee. Use
 * whereStr(...) (the shared nadata-language filter) instead.
 *
 * Memory contract (docs/native.md):
 * - inputs are borrowed: Java-side structures only need to stay
 *   referenced for the duration of the call (JNA guarantees this for
 *   by-reference Structure parameters);
 * - outputs are owned by the caller: the high-level wrapper reads
 *   them into Java objects and releases them via ndFreeValue
 *   immediately, so nothing crosses a GC boundary.
 */
public final class NadataLibrary {

    /** Status codes (NdStatusCode). */
    public static final int ND_OK = 0;
    public static final int ND_ERR_NOTFOUND = -1;
    public static final int ND_ERR_NAME = -2;
    public static final int ND_ERR_VALUE = -3;
    public static final int ND_ERR_TYPE = -4;
    public static final int ND_ERR_SYNTAX = -5;
    public static final int ND_ERR_IO = -6;
    public static final int ND_ERR_LIMIT = -7;
    public static final int ND_ERR_PERMISSION = -8;
    public static final int ND_ERR_RUNTIME = -9;
    public static final int ND_ERR_INVALID_ARG = -10;
    public static final int ND_ERR_STATE = -11;
    public static final int ND_ERR_MEMORY = -12;

    /** NdType tags. */
    public static final int ND_NULL = 0;
    public static final int ND_BOOL = 1;
    public static final int ND_INT64 = 2;
    public static final int ND_FLOAT64 = 3;
    public static final int ND_STR = 4;
    public static final int ND_ARRAY = 5;
    public static final int ND_DICT = 6;
    public static final int ND_INSTANCE = 7;

    /** The NdValue union (tag + union, 32 bytes — matches nadata.h). */
    public static class ValueUnion extends Union {
        public int boolean_;
        public long i64;
        public double f64;
        public Pointer str;
        public Array array;
        public Dict dict;
        public long instance_id;
    }

    public static class Array extends Structure {
        public Pointer items;    /* heap array of count NdValues */
        public long count;

        @Override
        protected java.util.List<String> getFieldOrder() {
            return java.util.Arrays.asList("items", "count");
        }
    }

    public static class Dict extends Structure {
        public Pointer keys;     /* heap array of count char*    */
        public Pointer values;   /* heap array of count NdValues */
        public long count;

        @Override
        protected java.util.List<String> getFieldOrder() {
            return java.util.Arrays.asList("keys", "values", "count");
        }
    }

    public static class NdValue extends Structure {
        public int tag;
        public ValueUnion as;

        @Override
        protected java.util.List<String> getFieldOrder() {
            return java.util.Arrays.asList("tag", "as");
        }
    }

    /** Function pointer types (all resolved by name at load time). */
    public interface Fn {
        /* Marker — the actual mappings live in Native.load below. */
    }

    private NadataLibrary() {
    }
}
