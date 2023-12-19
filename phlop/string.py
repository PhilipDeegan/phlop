#
#
#
#
#


def decode_bytes(input, as_format="utf-8"):
    if isinstance(input, str):
        return input
    if isinstance(input, bytes):
        return input.decode(as_format, errors="ignore")
    raise ValueError(f"phlop.string::decode_bytes unknown input type: {type(input)}")
