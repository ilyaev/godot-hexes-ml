for line in open("xor.py"):
    print(line.upper(), end='')


f = open("xor.py")
print(f.readlines())
f.close()
