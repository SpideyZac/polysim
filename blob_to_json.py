"""
Converts a binary track blob back into JSON format.

This script reverses the binary format produced by the JSON-to-binary
converter used by the simulation runtime.
"""

import json
import struct
import sys


def read_u32(file_obj):
    """
    Read an unsigned 32-bit integer.

    Args:
        file_obj: Binary file object.

    Returns:
        The decoded unsigned integer.
    """
    return struct.unpack("<I", file_obj.read(4))[0]


def read_i32(file_obj):
    """
    Read a signed 32-bit integer.

    Args:
        file_obj: Binary file object.

    Returns:
        The decoded signed integer.
    """
    return struct.unpack("<i", file_obj.read(4))[0]


def read_u8(file_obj):
    """
    Read an unsigned 8-bit integer.

    Args:
        file_obj: Binary file object.

    Returns:
        The decoded unsigned byte.
    """
    return struct.unpack("<B", file_obj.read(1))[0]


def read_f32(file_obj):
    """
    Read a 32-bit floating point value.

    Args:
        file_obj: Binary file object.

    Returns:
        The decoded float.
    """
    return struct.unpack("<f", file_obj.read(4))[0]


def read_vec3(file_obj):
    """
    Read a 3D vector.

    Args:
        file_obj: Binary file object.

    Returns:
        A list containing three float values.
    """
    return list(struct.unpack("<fff", file_obj.read(12)))


def read_vec_f32(file_obj):
    """
    Read a variable-length float vector.

    Data format:
        [count:u32][count × float]

    Args:
        file_obj: Binary file object.

    Returns:
        A flat list of floats.
    """
    count = read_u32(file_obj)
    return list(struct.unpack("<" + ("f" * count), file_obj.read(count * 4)))


def chunk(values, size):
    """
    Split a flat list into fixed-size chunks.

    Args:
        values: Input list.
        size: Number of elements per chunk.

    Returns:
        A nested list containing grouped elements.
    """
    return [values[index : index + size] for index in range(0, len(values), size)]


def main():
    """
    Convert a binary track definition into JSON.

    Command line arguments:
        argv[1]: Input binary file path.
        argv[2]: Output JSON file path.
    """
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    result = {}

    with open(input_file, "rb") as file_obj:
        collision_flat = read_vec_f32(file_obj)
        result["carCollisionShapeVertices"] = chunk(collision_flat, 3)

        result["carMassOffset"] = read_f32(file_obj)

        num_parts = read_u32(file_obj)
        parts = []

        for _ in range(num_parts):
            part = {}

            part["id"] = read_u32(file_obj)

            vertex_data = read_vec_f32(file_obj)

            vertices = {}
            grouped = chunk(vertex_data, 3)

            for index, vertex in enumerate(grouped):
                vertices[f"v{index}"] = vertex

            part["vertices"] = vertices

            has_detector = read_u8(file_obj)

            if has_detector:
                part["detector"] = {
                    "type": read_i32(file_obj),
                    "center": read_vec3(file_obj),
                    "size": read_vec3(file_obj),
                }

            has_offset = read_u8(file_obj)

            if has_offset:
                part["startOffset"] = read_vec3(file_obj)

            parts.append(part)

        result["trackParts"] = parts

    with open(output_file, "w", encoding="utf-8") as file_obj:
        json.dump(result, file_obj, indent=2)

    print(f"Converted {input_file} -> {output_file}")


if __name__ == "__main__":
    main()
