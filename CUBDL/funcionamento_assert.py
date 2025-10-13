def dividir(a, b):
    assert b != 0, "O denominador não pode ser zero"
    return a / b

print(dividir(10, 2))   # ok
print(dividir(10, 0))   # gera AssertionError