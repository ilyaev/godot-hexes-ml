from functools import reduce

regions = [{"population": 1}, {"population": 3}]

print(list(map(lambda x: x['population'], regions)))
