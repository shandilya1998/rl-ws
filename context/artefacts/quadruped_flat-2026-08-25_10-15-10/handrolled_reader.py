"""Minimal hand-rolled TFRecord + protobuf wire-format reader, used only to
cross-validate the tensorboard library's event_accumulator output for a
handful of records. Implements the exact scheme in the task prompt: uint64
LE length, masked crc32c of length, payload bytes, masked crc32c of payload.
Event.step is field 2 varint, Event.summary is field 5 (embedded message),
Summary.value is field 1 repeated (embedded message), Value.tag is field 1
(string), Value.simple_value is field 2 (float32, wire type 5).
"""
import struct
import sys

PATH = "/ws/IsaacLab/logs/rsl_rl/quadruped_flat/2026-08-25_10-15-10/events.out.tfevents.1787653075.auk6000.10589.0"

def read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos

def parse_fields(buf):
    """Yield (field_number, wire_type, value_bytes_or_int_or_float) for a
    top-level protobuf message given as bytes."""
    pos = 0
    n = len(buf)
    out = []
    while pos < n:
        tag, pos = read_varint(buf, pos)
        field_no = tag >> 3
        wire_type = tag & 0x7
        if wire_type == 0:  # varint
            val, pos = read_varint(buf, pos)
            out.append((field_no, wire_type, val))
        elif wire_type == 1:  # 64-bit
            val = buf[pos:pos+8]
            pos += 8
            out.append((field_no, wire_type, val))
        elif wire_type == 2:  # length-delimited
            length, pos = read_varint(buf, pos)
            val = buf[pos:pos+length]
            pos += length
            out.append((field_no, wire_type, val))
        elif wire_type == 5:  # 32-bit
            val = buf[pos:pos+4]
            pos += 4
            out.append((field_no, wire_type, val))
        else:
            raise ValueError(f"unsupported wire type {wire_type} at pos {pos}")
    return out

def parse_event(buf):
    step = None
    wall_time = None
    values = []  # (tag, simple_value)
    for field_no, wt, val in parse_fields(buf):
        if field_no == 1 and wt == 1:  # wall_time double
            wall_time = struct.unpack('<d', val)[0]
        elif field_no == 2 and wt == 0:  # step varint
            step = val
        elif field_no == 5 and wt == 2:  # summary message
            for f2, wt2, v2 in parse_fields(val):
                if f2 == 1 and wt2 == 2:  # Value message
                    tag = None
                    simple_value = None
                    for f3, wt3, v3 in parse_fields(v2):
                        if f3 == 1 and wt3 == 2:
                            tag = v3.decode('utf-8', errors='replace')
                        elif f3 == 2 and wt3 == 5:
                            simple_value = struct.unpack('<f', v3)[0]
                    if tag is not None:
                        values.append((tag, simple_value))
    return step, wall_time, values

def read_records(path, max_records=None):
    with open(path, 'rb') as f:
        count = 0
        while True:
            header = f.read(12)  # 8 byte length + 4 byte masked crc
            if len(header) < 12:
                break
            length = struct.unpack('<Q', header[:8])[0]
            data = f.read(length)
            f.read(4)  # masked crc of data
            yield data
            count += 1
            if max_records is not None and count >= max_records:
                break

if __name__ == "__main__":
    n_show = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    tags_seen = set()
    rows = []
    for i, rec in enumerate(read_records(PATH, max_records=None)):
        step, wall_time, values = parse_event(rec)
        for tag, sv in values:
            tags_seen.add(tag)
        if i < n_show:
            rows.append((i, step, wall_time, values))
    for r in rows:
        print(r)
    print("TOTAL_TAGS_SEEN:", len(tags_seen))
    for t in sorted(tags_seen):
        print("TAG:", t)
