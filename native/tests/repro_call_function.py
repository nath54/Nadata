# Bisect: which facade calls crash from the shim? (ctypes, raw bytes)
import ctypes
import faulthandler
import os
import struct

faulthandler.enable()

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["NADATA_HOME"] = os.getcwd()

dll = ctypes.PyDLL(os.path.join("native", "build", "win32", "nadata.dll"))
dll.nd_last_error.restype = ctypes.c_char_p


def nd_value(tag: int, payload: bytes) -> bytes:
    return struct.pack("<i4x", tag) + payload.ljust(24, b"\0")


def nd_instance(instance_id: int) -> bytes:
    return nd_value(7, struct.pack("<q", instance_id))


def nd_array(items_ptr: int, count: int) -> bytes:
    return nd_value(5, struct.pack("<Qq", items_ptr, count))


db = ctypes.c_void_p()
print("load:", dll.nd_load(b"./examples/ex0_simple", ctypes.byref(db)), flush=True)

tom = ctypes.c_int64(-1)
dll.nd_get_symbol_id.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
dll.nd_get_symbol_id(db, b"tom", ctypes.byref(tom))
print("tom id:", tom.value, flush=True)

items = (ctypes.c_char * 32)(*(nd_instance(tom.value)))
args_buf = ctypes.create_string_buffer(nd_array(ctypes.addressof(items), 1), 32)
out = ctypes.create_string_buffer(32)

dll.nd_call_function.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]
print("calling call_function is_adult...", flush=True)
rc = dll.nd_call_function(db, b"is_adult", ctypes.addressof(args_buf), None,
                          ctypes.addressof(out))
print("call_function rc:", rc, "err:", dll.nd_last_error(None), flush=True)
if rc == 0:
    print("out tag:", struct.unpack_from("<i", out.raw, 0)[0], flush=True)
