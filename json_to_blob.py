"""
Converts a track definition from JSON format to a binary blob format used by the simulation runtime.
"""

import json
import struct
import sys
import typing


def write_vec_f32(f: typing.BinaryIO, values: list):
    """Writes a list of floats to the file, prefixed by the count."""

    flat = []

    for v in values:
        if isinstance(v, (list, tuple)):
            for x in v:
                flat.append(float(x))
        else:
            flat.append(float(v))

    f.write(struct.pack("<I", len(flat)))
    f.write(struct.pack("<" + "f" * len(flat), *flat))


def write_vec3(f: typing.BinaryIO, v: list):
    """Writes a 3D vector to the file."""
    f.write(struct.pack("<fff", *v))


def main():
    """Main function to convert JSON to binary blob."""
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, encoding="utf-8") as fp:
        data = json.load(fp)

    with open(output_file, "wb") as f:
        # car collision vertices
        write_vec_f32(f, data["carCollisionShapeVertices"])

        # mass offset
        f.write(struct.pack("<f", data["carMassOffset"]))

        # track parts
        parts = data["trackParts"]
        f.write(struct.pack("<I", len(parts)))

        for p in parts:
            # id
            f.write(struct.pack("<I", p["id"]))

            # vertices
            write_vec_f32(f, p["vertices"].values())

            # detector
            detector = p.get("detector")
            if detector:
                f.write(struct.pack("<B", 1))
                f.write(struct.pack("<i", detector["type"]))
                write_vec3(f, detector["center"])
                write_vec3(f, detector["size"])
            else:
                f.write(struct.pack("<B", 0))

            # start offset
            offset = p.get("startOffset")
            if offset:
                f.write(struct.pack("<B", 1))
                write_vec3(f, offset)
            else:
                f.write(struct.pack("<B", 0))


if __name__ == "__main__":
    main()
