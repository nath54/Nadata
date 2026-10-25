# Cross-language benchmark results

- Generated: 2026-10-26 00:57:09
- Platform: Windows AMD64 (11)
- Python (facade + sqlite): 3.14.7
- C++: reference binding over the C shim (embedded runtime)

ops/sec, higher is better. C++ and Python run the full nadata
engine (typed attributes, expression evaluation, embedded
interpreter); SQLite is raw row I/O with no engine semantics —
see benchmarks/README.md for the fairness caveats.

### load

| Implementation | ops | ms | ops/sec |
|---|---|---|---|
| C++ (embedded runtime) | 20 | 327.9 | 61 |
| Python (facade) | 20 | 167.8 | 119 |
| SQLite (raw storage reference) | 20 | 11.7 | 1715 |

### get_attribute

| Implementation | ops | ms | ops/sec |
|---|---|---|---|
| C++ (embedded runtime) | 10000 | 18.2 | 550191 |
| Python (facade) | 10000 | 8.4 | 1195900 |
| SQLite (raw storage reference) | 10000 | 15.6 | 639754 |

### set_attribute

| Implementation | ops | ms | ops/sec |
|---|---|---|---|
| C++ (embedded runtime) | 10000 | 44.3 | 225817 |
| Python (facade) | 10000 | 33.1 | 302361 |
| SQLite (raw storage reference) | 10000 | 11.4 | 876363 |

### query_where_count

| Implementation | ops | ms | ops/sec |
|---|---|---|---|
| C++ (embedded runtime) | 1000 | 44.9 | 22283 |
| Python (facade) | 1000 | 44.7 | 22369 |
| SQLite (raw storage reference) | 1000 | 1.5 | 664364 |

### create_instance

| Implementation | ops | ms | ops/sec |
|---|---|---|---|
| C++ (embedded runtime) | 1000 | 9.5 | 105471 |
| Python (facade) | 1000 | 7.3 | 137421 |
| SQLite (raw storage reference) | 1000 | 1.3 | 798467 |

### call_function

| Implementation | ops | ms | ops/sec |
|---|---|---|---|
| C++ (embedded runtime) | 10000 | 532.0 | 18796 |
| Python (facade) | 10000 | 480.2 | 20824 |
| SQLite (raw storage reference) | 10000 | 16.9 | 591377 |

### Memory (peak RSS)

| Implementation | peak MB | delta MB |
|---|---|---|
| C++ (embedded runtime) | 25.6 | 20.2 |
| Python (facade) | 35.0 | 0.4 |
| SQLite (raw storage reference) | 35.5 | 0.0 |

### Output verification

- C++ (embedded runtime): 5 checks, all ok (exit 0)
- Python (facade): 5 checks, all ok (exit 0)
- SQLite (raw storage reference): 5 checks, all ok (exit 0)
