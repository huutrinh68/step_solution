import json
from pprint import pprint

with open('dis_info_04.txt') as f:
    data = json.load(f)
print(data)