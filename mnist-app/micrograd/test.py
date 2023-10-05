
from micrograd.engine import Value

a = Value(3)
b = Value(4)

a = a + b

a.backward()

print(a.grad)