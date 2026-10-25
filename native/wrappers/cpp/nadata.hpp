/*
 * nadata.hpp — C++17 RAII wrapper over the nadata C ABI (nadata.h).
 *
 * Header-only, zero dependencies beyond the C standard library. Maps
 * the ABI 1:1 onto the Python facade (Choice A), same as nadata.h:
 *
 *   nadata::Database db = nadata::Database::load("./examples/ex0_simple");
 *   nadata::Query q = db.query("Human");
 *   q.where_str("h", "h.name == \"Tom\"");
 *   int64_t id = q.first_id().value();
 *
 * Ownership rules (mirroring the ABI contract):
 *   - Database / Query own their handles (nd_close / nd_query_free).
 *   - Value owns its NdValue (recursive nd_free_value). Nested arrays
 *     and dicts built by this wrapper are allocated with malloc/free —
 *     the allocator the ABI's nd_free_value uses.
 *   - Inputs are borrowed by the shim: Value arguments may be released
 *     right after a call.
 *
 * Errors: every failing ABI call throws nadata::Error carrying the
 * NdStatusCode and the shim's thread-local message.
 *
 * Threading: safe to call from any thread; the database executes one
 * call at a time (see nadata.h concurrency contract).
 */

#ifndef NADATA_WRAPPER_HPP
#define NADATA_WRAPPER_HPP

/* Import Modules */
#include "nadata.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nadata {

/* ------------------------------------------------------------------ */
/* Error                                                               */
/* ------------------------------------------------------------------ */

/*
 * Raised by every fallible wrapper call. `status` is the ABI status,
 * `what()` is the shim's human-readable message for that call.
 */
class Error : public std::runtime_error {
public:
    Error(NdStatusCode status, const std::string& message)
        : std::runtime_error(message), status_(status) {}

    NdStatusCode status() const noexcept {
        return status_;
    }

private:
    NdStatusCode status_;
};

/* Throws Error when `status` is negative. */
inline void require(NdStatusCode status, const char* context,
                    NdDb* db = nullptr)
{
    if (status == ND_OK) {
        return;
    }

    const char* message = nd_last_error(db);
    std::string text = context;

    if (message != NULL) {
        text += ": ";
        text += message;
    }

    throw Error(status, text);
}

/* ------------------------------------------------------------------ */
/* Value                                                               */
/* ------------------------------------------------------------------ */

/*
 * Owns an NdValue. Default-constructed = ND_NULL, which every ABI
 * entry accepting an optional value treats as "absent" — so
 * `db.call_function("f")` and `db.call_function("f", Value::array(...))`
 * share one signature.
 */
class Value {
public:
    /* ------------------------- construction ------------------------ */

    Value() {
        std::memset(&raw_, 0, sizeof(raw_));
        raw_.tag = ND_NULL;
    }

    explicit Value(NdValue raw) : raw_() {   /* takes ownership */
        std::memset(&raw_, 0, sizeof(raw_));
        std::swap(raw_, raw);
    }

    Value(Value&& other) noexcept : Value() {
        std::swap(raw_, other.raw_);
    }

    Value& operator=(Value&& other) noexcept {
        if (this != &other) {
            reset();
            std::swap(raw_, other.raw_);
        }

        return *this;
    }

    /*
     * Deep-copying copy semantics: values are immutable data, and
     * copyability keeps vector<>/initializer_list construction and
     * at()/key_at() ergonomic. Moves stay O(1).
     */
    Value(const Value& other) : raw_(clone_raw(other.raw_)) {}

    Value& operator=(const Value& other) {
        if (this != &other) {
            reset();
            raw_ = clone_raw(other.raw_);
        }

        return *this;
    }

    ~Value() {
        reset();
    }

    /* --------------------------- builders -------------------------- */

    static Value null() {
        return Value();
    }

    static Value boolean(bool value) {
        NdValue raw;
        std::memset(&raw, 0, sizeof(raw));
        raw.tag = ND_BOOL;
        raw.as.boolean = value ? 1 : 0;

        return Value(raw);
    }

    static Value integer(int64_t value) {
        NdValue raw;
        std::memset(&raw, 0, sizeof(raw));
        raw.tag = ND_INT64;
        raw.as.i64 = value;

        return Value(raw);
    }

    static Value floating(double value) {
        NdValue raw;
        std::memset(&raw, 0, sizeof(raw));
        raw.tag = ND_FLOAT64;
        raw.as.f64 = value;

        return Value(raw);
    }

    static Value str(const std::string& text) {
        NdValue raw;
        std::memset(&raw, 0, sizeof(raw));
        raw.tag = ND_STR;
        raw.as.str = copy_string(text);

        return Value(raw);
    }

    static Value instance(int64_t entity_id) {
        NdValue raw;
        std::memset(&raw, 0, sizeof(raw));
        raw.tag = ND_INSTANCE;
        raw.as.instance_id = entity_id;

        return Value(raw);
    }

    /* Deep-copies `items` into a malloc'd ND_ARRAY. */
    static Value array(const std::vector<Value>& items) {
        NdValue raw;
        std::memset(&raw, 0, sizeof(raw));
        raw.tag = ND_ARRAY;
        raw.as.array.count = static_cast<int64_t>(items.size());
        raw.as.array.items = items.empty()
            ? nullptr
            : static_cast<NdValue*>(std::malloc(
                  items.size() * sizeof(NdValue)));

        if (raw.as.array.items == nullptr && !items.empty()) {
            throw Error(ND_ERR_MEMORY, "out of memory building ND_ARRAY");
        }

        for (size_t i = 0; i < items.size(); i++) {
            raw.as.array.items[i] = clone_raw(items[i].raw_);
        }

        return Value(raw);
    }

    /* Deep-copies `entries` into a malloc'd ND_DICT. */
    static Value dict(
        const std::vector<std::pair<std::string, Value>>& entries) {
        NdValue raw;
        std::memset(&raw, 0, sizeof(raw));
        raw.tag = ND_DICT;
        raw.as.dict.count = static_cast<int64_t>(entries.size());
        raw.as.dict.keys = entries.empty()
            ? nullptr
            : static_cast<char**>(std::malloc(
                  entries.size() * sizeof(char*)));
        raw.as.dict.values = entries.empty()
            ? nullptr
            : static_cast<NdValue*>(std::malloc(
                  entries.size() * sizeof(NdValue)));

        if (entries.empty()) {
            return Value(raw);
        }

        if (raw.as.dict.keys == nullptr || raw.as.dict.values == nullptr) {
            nd_free_value(&raw);
            throw Error(ND_ERR_MEMORY, "out of memory building ND_DICT");
        }

        for (size_t i = 0; i < entries.size(); i++) {
            raw.as.dict.keys[i] = copy_string(entries[i].first);
            raw.as.dict.values[i] = clone_raw(entries[i].second.raw_);
        }

        return Value(raw);
    }

    /* Convenience: single-pair dict (attributes with one field). */
    static Value dict(const std::string& key, Value value) {
        return dict({{key, std::move(value)}});
    }

    /* --------------------------- accessors ------------------------- */

    NdType type() const noexcept {
        return raw_.tag;
    }

    bool is_null() const noexcept {
        return raw_.tag == ND_NULL;
    }

    bool as_bool() const {
        assert_type(ND_BOOL, "as_bool");

        return raw_.as.boolean != 0;
    }

    int64_t as_int() const {
        assert_type(ND_INT64, "as_int");

        return raw_.as.i64;
    }

    double as_float() const {
        assert_type(ND_FLOAT64, "as_float");

        return raw_.as.f64;
    }

    /* Valid while this Value is alive. */
    const char* as_str() const {
        assert_type(ND_STR, "as_str");

        return raw_.as.str != nullptr ? raw_.as.str : "";
    }

    int64_t as_instance_id() const {
        assert_type(ND_INSTANCE, "as_instance_id");

        return raw_.as.instance_id;
    }

    size_t size() const {
        if (raw_.tag == ND_ARRAY) {
            return static_cast<size_t>(raw_.as.array.count);
        }

        if (raw_.tag == ND_DICT) {
            return static_cast<size_t>(raw_.as.dict.count);
        }

        assert_type(ND_ARRAY, "size");

        return 0;
    }

    /* Deep copy of one array element. */
    Value at(size_t index) const {
        assert_type(ND_ARRAY, "at");

        if (static_cast<int64_t>(index) >= raw_.as.array.count) {
            throw Error(ND_ERR_INVALID_ARG, "array index out of range");
        }

        return Value(clone_raw(raw_.as.array.items[index]));
    }

    /* Deep copy of dict key/value by index. */
    std::string key_at(size_t index) const {
        assert_type(ND_DICT, "key_at");

        if (static_cast<int64_t>(index) >= raw_.as.dict.count) {
            throw Error(ND_ERR_INVALID_ARG, "dict index out of range");
        }

        return raw_.as.dict.keys[index];
    }

    Value value_at(size_t index) const {
        assert_type(ND_DICT, "value_at");

        if (static_cast<int64_t>(index) >= raw_.as.dict.count) {
            throw Error(ND_ERR_INVALID_ARG, "dict index out of range");
        }

        return Value(clone_raw(raw_.as.dict.values[index]));
    }

    /* Borrowed view for passing as an INPUT argument. The shim copies
     * during the call, so the Value only needs to outlive the call. */
    const NdValue* raw() const noexcept {
        return &raw_;
    }

private:
    NdValue raw_;

    void reset() {
        if (raw_.tag != ND_NULL) {
            nd_free_value(&raw_);
        }

        std::memset(&raw_, 0, sizeof(raw_));
        raw_.tag = ND_NULL;
    }

    void assert_type(NdType expected, const char* accessor) const {
        if (raw_.tag != expected) {
            throw Error(ND_ERR_TYPE,
                        std::string(accessor) + ": wrong value type");
        }
    }

    /* malloc'd string — the allocator nd_free_value releases with. */
    static char* copy_string(const std::string& text) {
        char* copy = static_cast<char*>(std::malloc(text.size() + 1));

        if (copy == nullptr) {
            throw Error(ND_ERR_MEMORY, "out of memory copying string");
        }

        std::memcpy(copy, text.c_str(), text.size() + 1);

        return copy;
    }

    /* Deep copy of an NdValue tree using the ABI's allocator. */
    static NdValue clone_raw(const NdValue& source) {
        NdValue copy;
        std::memset(&copy, 0, sizeof(copy));
        copy.tag = source.tag;

        switch (source.tag) {
            case ND_STR:
                copy.as.str = copy_string(source.as.str != nullptr
                    ? source.as.str : "");
                break;

            case ND_ARRAY:
                copy.as.array.count = source.as.array.count;
                copy.as.array.items = source.as.array.count == 0
                    ? nullptr
                    : static_cast<NdValue*>(std::malloc(
                          source.as.array.count * sizeof(NdValue)));

                if (copy.as.array.items == nullptr
                    && source.as.array.count > 0) {
                    throw Error(ND_ERR_MEMORY,
                                "out of memory cloning array");
                }

                for (int64_t i = 0; i < source.as.array.count; i++) {
                    copy.as.array.items[i] =
                        clone_raw(source.as.array.items[i]);
                }

                break;

            case ND_DICT:
                copy.as.dict.count = source.as.dict.count;
                copy.as.dict.keys = source.as.dict.count == 0
                    ? nullptr
                    : static_cast<char**>(std::malloc(
                          source.as.dict.count * sizeof(char*)));
                copy.as.dict.values = source.as.dict.count == 0
                    ? nullptr
                    : static_cast<NdValue*>(std::malloc(
                          source.as.dict.count * sizeof(NdValue)));

                if ((copy.as.dict.keys == nullptr
                     || copy.as.dict.values == nullptr)
                    && source.as.dict.count > 0) {
                    throw Error(ND_ERR_MEMORY,
                                "out of memory cloning dict");
                }

                for (int64_t i = 0; i < source.as.dict.count; i++) {
                    copy.as.dict.keys[i] =
                        copy_string(source.as.dict.keys[i]);
                    copy.as.dict.values[i] =
                        clone_raw(source.as.dict.values[i]);
                }

                break;

            default:
                copy.as = source.as;   /* scalar payload */
                break;
        }

        return copy;
    }
};

/* ------------------------------------------------------------------ */
/* Query                                                               */
/* ------------------------------------------------------------------ */

/*
 * Owns an NdQuery handle. ABI query methods rebind the handle in
 * place, so filters chain by reference:
 *
 *   Query q = db.query("Human");
 *   q.where_str("h", "h.age(DateTime.now()).year >= 18")
 *    .order_by("name").limit(10);
 */
class Query {
public:
    explicit Query(NdQuery* handle) : handle_(handle) {}

    Query(Query&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Query& operator=(Query&& other) noexcept {
        if (this != &other) {
            release();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }

        return *this;
    }

    Query(const Query&) = delete;
    Query& operator=(const Query&) = delete;

    ~Query() {
        release();
    }

    /* --------------------------- filters --------------------------- */

    /* Chainable filters return Query&& so they work on temporaries
     * (db.query("H").where_str(...)) and lvalues alike; the ABI rebinds
     * the handle in place. */
    Query&& where_str(const std::string& var, const std::string& expr) {
        require(nd_query_where_str(handle_, var.c_str(), expr.c_str()),
                "nd_query_where_str");

        return std::move(*this);
    }

    Query&& where_cb(NdPredicateFn predicate, void* user_data) {
        require(nd_query_where_cb(handle_, predicate, user_data),
                "nd_query_where_cb");

        return std::move(*this);
    }

    Query&& order_by(const std::string& attribute, bool descending = false) {
        require(nd_query_order_by(handle_, attribute.c_str(),
                                  descending ? 1 : 0),
                "nd_query_order_by");

        return std::move(*this);
    }

    Query&& limit(int64_t n) {
        require(nd_query_limit(handle_, n), "nd_query_limit");

        return std::move(*this);
    }

    /* ---------------------------- reads ---------------------------- */

    int64_t count() const {
        int64_t out = 0;
        require(nd_query_count(handle_, &out), "nd_query_count");

        return out;
    }

    bool exists() const {
        int32_t out = 0;
        require(nd_query_exists(handle_, &out), "nd_query_exists");

        return out != 0;
    }

    /* Empty set -> nullopt (the ABI's explicit NOTFOUND contract). */
    std::optional<int64_t> first_id() {
        int64_t out = 0;
        NdStatusCode status = nd_query_first_id(handle_, &out);

        if (status == ND_ERR_NOTFOUND) {
            return std::nullopt;
        }

        require(status, "nd_query_first_id");

        return out;
    }

    std::optional<int64_t> last_id() {
        int64_t out = 0;
        NdStatusCode status = nd_query_last_id(handle_, &out);

        if (status == ND_ERR_NOTFOUND) {
            return std::nullopt;
        }

        require(status, "nd_query_last_id");

        return out;
    }

    Value pluck(const std::string& attribute) {
        NdValue out;
        std::memset(&out, 0, sizeof(out));

        require(nd_query_pluck(handle_, attribute.c_str(), &out),
                "nd_query_pluck");

        return Value(out);
    }

    Value sum_of(const std::string& attribute) {
        NdValue out;
        std::memset(&out, 0, sizeof(out));

        require(nd_query_sum_of(handle_, attribute.c_str(), &out),
                "nd_query_sum_of");

        return Value(out);
    }

    Value min_of(const std::string& attribute) {
        NdValue out;
        std::memset(&out, 0, sizeof(out));

        require(nd_query_min_of(handle_, attribute.c_str(), &out),
                "nd_query_min_of");

        return Value(out);
    }

    Value max_of(const std::string& attribute) {
        NdValue out;
        std::memset(&out, 0, sizeof(out));

        require(nd_query_max_of(handle_, attribute.c_str(), &out),
                "nd_query_max_of");

        return Value(out);
    }

    double average_of(const std::string& attribute) {
        double out = 0.0;

        require(nd_query_average_of(handle_, attribute.c_str(), &out),
                "nd_query_average_of");

        return out;
    }

    std::vector<int64_t> to_list_ids() {
        NdValue out;
        std::memset(&out, 0, sizeof(out));

        require(nd_query_to_list_ids(handle_, &out),
                "nd_query_to_list_ids");

        Value ids(out);
        std::vector<int64_t> result;

        for (size_t i = 0; i < ids.size(); i++) {
            result.push_back(ids.at(i).as_instance_id());
        }

        return result;
    }

private:
    NdQuery* handle_;

    void release() {
        if (handle_ != nullptr) {
            nd_query_free(handle_);
            handle_ = nullptr;
        }
    }
};

/* ------------------------------------------------------------------ */
/* Runtime info                                                        */
/* ------------------------------------------------------------------ */

/* JSON build/interpreter info; free function (no db needed). */
inline std::string runtime_info() {
    char* info = nd_runtime_info();

    if (info == nullptr) {
        return "";
    }

    std::string result(info);
    nd_free_str(info);

    return result;
}

/* ------------------------------------------------------------------ */
/* Database                                                            */
/* ------------------------------------------------------------------ */

/*
 * Owns an NdDb handle; every method maps 1:1 onto nadata.api.Nadata.
 * All failing calls throw nadata::Error.
 */
class Database {
public:
    /* -------------------------- lifecycle -------------------------- */

    static Database load(const std::string& path) {
        NdDb* handle = nullptr;

        require(nd_load(path.c_str(), &handle), "nd_load");

        return Database(handle);
    }

    static Database open(const std::string& storage_path) {
        NdDb* handle = nullptr;

        require(nd_open(storage_path.c_str(), &handle), "nd_open");

        return Database(handle);
    }

    static Database open_folder(const std::string& folder_path,
                                const std::string& db_path = "",
                                bool run_auto = true,
                                bool force_init = false) {
        NdDb* handle = nullptr;

        require(nd_open_folder(
                    folder_path.c_str(),
                    db_path.empty() ? nullptr : db_path.c_str(),
                    run_auto ? 1 : 0, force_init ? 1 : 0, &handle),
                "nd_open_folder");

        return Database(handle);
    }

    Database(Database&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Database& operator=(Database&& other) noexcept {
        if (this != &other) {
            release();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }

        return *this;
    }

    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;

    ~Database() {
        release();
    }

    /* ------------------------ introspection ------------------------ */

    std::vector<std::string> struct_names() const {
        return string_list(nd_struct_names, "nd_struct_names");
    }

    std::vector<std::string> function_names() const {
        return string_list(nd_function_names, "nd_function_names");
    }

    std::vector<std::string> symbol_names() const {
        return string_list(nd_symbol_names, "nd_symbol_names");
    }

    std::string summary() const {
        char* out = nullptr;

        require(nd_summary(handle_, &out), "nd_summary", handle_);

        std::string result(out != nullptr ? out : "");
        nd_free_str(out);

        return result;
    }

    int64_t count(const std::string& struct_name) const {
        int64_t out = 0;

        require(nd_count(handle_, struct_name.c_str(), &out),
                "nd_count", handle_);

        return out;
    }

    int64_t total_instance_count() const {
        int64_t out = 0;

        require(nd_total_instance_count(handle_, &out),
                "nd_total_instance_count", handle_);

        return out;
    }

    /* -------------------- symbols + attributes --------------------- */

    int64_t get_symbol_id(const std::string& symbol) const {
        int64_t out = 0;

        require(nd_get_symbol_id(handle_, symbol.c_str(), &out),
                "nd_get_symbol_id", handle_);

        return out;
    }

    Value get_attribute(const std::string& symbol,
                        const std::string& name) const {
        NdValue out;
        std::memset(&out, 0, sizeof(out));

        require(nd_get_attribute(handle_, symbol.c_str(), name.c_str(),
                                 &out),
                "nd_get_attribute", handle_);

        return Value(out);
    }

    void set_attribute(const std::string& symbol, const std::string& name,
                       const Value& value) {
        require(nd_set_attribute(handle_, symbol.c_str(), name.c_str(),
                                 value.raw()),
                "nd_set_attribute", handle_);
    }

    /* `attributes` may be Value::null() (no fields) or a dict. */
    int64_t create_instance(const std::string& struct_name,
                            const Value& attributes,
                            const std::string& symbol = "") const {
        int64_t out = 0;

        require(nd_create_instance(
                    handle_, struct_name.c_str(),
                    attributes.is_null() ? nullptr : attributes.raw(),
                    symbol.empty() ? nullptr : symbol.c_str(), &out),
                "nd_create_instance", handle_);

        return out;
    }

    int64_t create_instance(
        const std::string& struct_name,
        const std::vector<std::pair<std::string, Value>>& attributes,
        const std::string& symbol = "") const {
        return create_instance(struct_name, Value::dict(attributes),
                               symbol);
    }

    /* --------------------------- calls ----------------------------- */

    Value call_function(const std::string& name,
                        const Value& args = Value(),
                        const Value& kwargs = Value()) const {
        NdValue out;
        std::memset(&out, 0, sizeof(out));

        require(nd_call_function(
                    handle_, name.c_str(),
                    args.is_null() ? nullptr : args.raw(),
                    kwargs.is_null() ? nullptr : kwargs.raw(), &out),
                "nd_call_function", handle_);

        return Value(out);
    }

    Value call_method(const std::string& symbol, const std::string& name,
                      const Value& args = Value(),
                      const Value& kwargs = Value()) const {
        NdValue out;
        std::memset(&out, 0, sizeof(out));

        require(nd_call_method(
                    handle_, symbol.c_str(), name.c_str(),
                    args.is_null() ? nullptr : args.raw(),
                    kwargs.is_null() ? nullptr : kwargs.raw(), &out),
                "nd_call_method", handle_);

        return Value(out);
    }

    /* ---------------------------- query ---------------------------- */

    Query query(const std::string& struct_name) const {
        NdQuery* handle = nullptr;

        require(nd_query_new(handle_, struct_name.c_str(), &handle),
                "nd_query_new", handle_);

        return Query(handle);
    }

    /* -------------------------- persistence ------------------------ */

    void dump(const std::string& storage_path) const {
        require(nd_dump(handle_, storage_path.c_str()), "nd_dump",
                handle_);
    }

    int64_t save_changes(const std::string& storage_path) const {
        int64_t docs = 0;

        require(nd_save_changes(handle_, storage_path.c_str(), &docs),
                "nd_save_changes", handle_);

        return docs;
    }

    std::pair<int64_t, int64_t> compact(
        const std::string& output_path = "") const {
        int64_t before = 0;
        int64_t after = 0;

        require(nd_compact(handle_,
                           output_path.empty()
                               ? nullptr : output_path.c_str(),
                           &before, &after),
                "nd_compact", handle_);

        return {before, after};
    }

    void set_loop_limit(int64_t limit) const {
        require(nd_set_loop_limit(handle_, limit), "nd_set_loop_limit",
                handle_);
    }

    /* ------------------------ code execution ----------------------- */

    void execute_schema(const std::string& schema_text) const {
        require(nd_execute_schema(handle_, schema_text.c_str()),
                "nd_execute_schema", handle_);
    }

    void execute_data(const std::string& data_text) const {
        require(nd_execute_data(handle_, data_text.c_str()),
                "nd_execute_data", handle_);
    }

    void execute(const std::string& text) const {
        require(nd_execute(handle_, text.c_str()), "nd_execute", handle_);
    }

    void execute_script_file(const std::string& file_path,
                             const Value& params = Value()) const {
        require(nd_execute_script_file(
                    handle_, file_path.c_str(),
                    params.is_null() ? nullptr : params.raw()),
                "nd_execute_script_file", handle_);
    }

private:
    NdDb* handle_ = nullptr;

    explicit Database(NdDb* handle) : handle_(handle) {}

    void release() {
        if (handle_ != nullptr) {
            nd_close(handle_);
            handle_ = nullptr;
        }
    }

    /*
     * Shared body for the three name-list calls: fills an ND_ARRAY of
     * ND_STR and converts it to std::vector<std::string>.
     */
    template <typename AbiCall>
    std::vector<std::string> string_list(AbiCall call,
                                         const char* context) const {
        NdValue out;
        std::memset(&out, 0, sizeof(out));

        require(call(handle_, &out), context, handle_);

        Value list(out);

        if (list.is_null()) {
            return {};
        }

        std::vector<std::string> names;

        for (size_t i = 0; i < list.size(); i++) {
            names.push_back(list.at(i).as_str());
        }

        return names;
    }
};

} /* namespace nadata */

#endif /* NADATA_WRAPPER_HPP */
