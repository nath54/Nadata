# Isolate the call_method crash via ctypes, building NdValues from raw
# bytes (the union field name `as` is a Python keyword).
import ctypes
import faulthandler
import os
import struct

faulthandler.enable()

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["NADATA_HOME"] = os.getcwd()

dll = ctypes.WinDLL(os.path.join("native", "build", "win32", "nadata.dll"))
dll.nd_last_error.restype = ctypes.c_char_p


def nd_value(tag: int, payload: bytes) -> bytes:
    """Builds a 32-byte NdValue: int32 tag + 4 pad + 24-byte union."""
    return struct.pack("<i4x", tag) + payload.ljust(24, b"\0")


def nd_instance(instance_id: int) -> bytes:
    """tag=ND_INSTANCE(7), union.instance_id at offset 8."""
    return nd_value(7, struct.pack("<q", instance_id).ljust(24, b"\0"))


def nd_array(items_ptr: int, count: int) -> bytes:
    """tag=ND_ARRAY(5), union.array = {ptr, count} at offset 8."""
    return nd_value(5, struct.pack("<Qq", items_ptr, count))


db = ctypes.c_void_p()
rc = dll.nd_load(b"./examples/ex0_simple", ctypes.byref(db))
print("load:", rc)
assert rc == 0

tom = ctypes.c_int64(-1)
dll.nd_get_symbol_id.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
rc = dll.nd_get_symbol_id(db, b"tom", ctypes.byref(tom))
print("tom:", rc, tom.value)

# args: ND_ARRAY of 1 ND_INSTANCE
items = (ctypes.c_char * 32)(*(nd_instance(tom.value)))
args_buf = ctypes.create_string_buffer(
    nd_array(ctypes.addressof(items), 1), 32
)

out = ctypes.create_string_buffer(32)

dll.nd_call_method.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]
print("calling call_method...")
rc = dll.nd_call_method(db, b"jean", b"older", ctypes.addressof(args_buf),
                        None, ctypes.addressof(out))
print("call_method rc:", rc, "err:", dll.nd_last_error(None))
if rc == 0:
    tag = struct.unpack_from("<i", out.raw, 0)[0]
    print("out tag:", tag)
