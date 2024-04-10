import numpy as np

# Crear una matriz de ejemplo
matriz = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Mostrar la matriz
print("Matriz:")
print(matriz)

# Forma de la matriz (número de filas y columnas)
print("\nForma de la matriz (número de filas y columnas):", matriz.shape)

# Número total de elementos en la matriz
print("Número total de elementos en la matriz:", matriz.size)

# Número de dimensiones de la matriz
print("Número de dimensiones de la matriz:", matriz.ndim)

# Tipo de datos de los elementos de la matriz
print("Tipo de datos de los elementos de la matriz:", matriz.dtype)

# Tamaño de los elementos de la matriz en bytes
print("Tamaño de los elementos de la matriz en bytes:", matriz.itemsize)

# Suma de todos los elementos de la matriz
print("Suma de todos los elementos de la matriz:", np.sum(matriz))

# Valor máximo y su posición en la matriz
print("Valor máximo de la matriz:", np.max(matriz))
print("Posición del valor máximo:", np.argmax(matriz))

# Valor mínimo y su posición en la matriz
print("Valor mínimo de la matriz:", np.min(matriz))
print("Posición del valor mínimo:", np.argmin(matriz))

# Transpuesta de la matriz
print("Transpuesta de la matriz:")
print(np.transpose(matriz))

# Determinante de la matriz (solo para matrices cuadradas)
if matriz.shape[0] == matriz.shape[1]:
    print("Determinante de la matriz:", np.linalg.det(matriz))
else:
    print("La matriz no es cuadrada, por lo tanto no se puede calcular el determinante.")
