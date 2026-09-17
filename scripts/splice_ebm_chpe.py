#!/usr/bin/env python3
"""Zero-copy sector splice into final .chpe binary allocation file."""
import argparse
import os
import struct
import sys

HEADER_SIZE = 4096
MAGIC = b"CHPE"

def main():
    parser = argparse.ArgumentParser(description="Zero-copy sector splice into .chpe binary")
    parser.add_argument("--layer-count", type=int, default=36, help="Number of layer slices to splice (default 36)")
    parser.add_argument("--layers-dir", default="db/layers", help="Directory containing layer slices")
    parser.add_argument("--out-file", default="models/qwen2.5-3b-ebm.chpe", help="Target .chpe path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    # 1. Verify all slices exist and are 16 KiB aligned
    for l in range(args.layer_count):
        slice_path = os.path.join(args.layers_dir, f"layer_{l}.chpe")
        if not os.path.exists(slice_path):
            print(f"Missing slice: {slice_path}", file=sys.stderr)
            sys.exit(1)
        size = os.path.getsize(slice_path)
        if size % 16384 != 0:
            print(f"Slice {slice_path} not 16 KiB sector aligned (size={size})", file=sys.stderr)
            sys.exit(1)

    # 2. Open destination and write header
    with open(args.out_file, "wb") as out_f:
        header = bytearray(HEADER_SIZE)
        header[0:4] = MAGIC
        struct.pack_into("<I", header, 4, 1) # version 1
        struct.pack_into("<I", header, 8, args.layer_count)
        out_f.write(header)

    out_fd = os.open(args.out_file, os.O_WRONLY)
    os.lseek(out_fd, 0, os.SEEK_END)
    try:
        total_spliced = 0
        for l in range(args.layer_count):
            slice_path = os.path.join(args.layers_dir, f"layer_{l}.chpe")
            in_fd = os.open(slice_path, os.O_RDONLY)
            size = os.path.getsize(slice_path)
            try:
                # Kernel zero-copy pipe via sendfile
                remaining = size
                while remaining > 0:
                    sent = os.sendfile(out_fd, in_fd, None, remaining)
                    if sent == 0:
                        break
                    remaining -= sent
                    total_spliced += sent
            finally:
                os.close(in_fd)
    finally:
        os.close(out_fd)

    print(f"Successfully spliced {args.layer_count} layers ({total_spliced} bytes) into {args.out_file}")
    sys.exit(0)

if __name__ == "__main__":
    main()
