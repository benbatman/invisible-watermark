import json

with open("test_digi_inno.json") as f:
    data = json.load(f)

json_str = json.dumps(data)
byte_str = json_str.encode("utf-8")
print(byte_str)

bit_str = "".join(format(byte, "08b") for byte in byte_str)
print(bit_str)
print(len(bit_str))

bytes_list = [int(bit_str[i : i + 8], 2) for i in range(0, len(bit_str), 8)]
byte_str = bytes(bytes_list)
json_str = byte_str.decode("utf-8")
print(json_str)
