import tpack.test as tt
import math
import sys
import collections
# print(sys.path)
# print(dir(math))


print(dir(tt))

tt.print_test()

tmp = tt.ModelTest()
print(tmp)

# print(dir(collections))

RowType = collections.namedtuple('RowType', ('id', 'name', 'value'))

rows = set()

for i in range(1):
    row = RowType(id=i, name="First Name {}".format(i),
                  value="first_name " + str(i))
    print(row)
    rows.add(row)

print(len(rows))
