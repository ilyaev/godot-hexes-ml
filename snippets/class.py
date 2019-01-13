class Parrot:

    species = 'Bird'
    name = ''
    age = -1

    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.a = 'b'

    def sing(self, song):
        return "{} sings {}".format(self.name, song)


obj = Parrot('One', 12)

print(obj.name, obj.age, obj.species)
print(obj.sing('Smoke On The Water'))
print(obj.__dict__)
