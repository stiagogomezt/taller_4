import numpy as np

# Crear un arreglo unidimensional
arr1d = np.array([1, 2, 3, 4, 5])
print("Arreglo unidimensional:")
print(arr1d)

# Crear un arreglo bidimensional
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\nArreglo bidimensional:")
print(arr2d)

# Operaciones básicas con arreglos
print("\nOperaciones básicas con arreglos:")
print("Suma:", arr1d + 2)
print("Resta:", arr1d - 2)
print("Multiplicación:", arr1d * 2)
print("División:", arr1d / 2)

# Funciones matemáticas
print("\nFunciones matemáticas:")
print("Seno:", np.sin(arr1d))
print("Coseno:", np.cos(arr1d))
print("Exponencial:", np.exp(arr1d))
print("Logaritmo natural:", np.log(arr1d))

# Crear un arreglo de números complejos
arr_complex = np.array([1 + 2j, 3 - 4j, 5 + 6j])
print("\nArreglo de números complejos:")
print(arr_complex)

# Operaciones con números complejos
print("\nOperaciones con números complejos:")
print("Parte real:", np.real(arr_complex))
print("Parte imaginaria:", np.imag(arr_complex))
print("Magnitud:", np.abs(arr_complex))

# Álgebra lineal
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("\nÁlgebra lineal:")
print("Producto punto (escalar):", np.dot(A, B))
print("Producto de matrices:", np.matmul(A, B))
print("Determinante de A:", np.linalg.det(A))
print("Inversa de A:", np.linalg.inv(A))

# Estadísticas básicas
arr_stats = np.array([[1, 2, 3], [4, 5, 6]])
print("\nEstadísticas básicas:")
print("Media:", np.mean(arr_stats))
print("Mediana:", np.median(arr_stats))
print("Desviación estándar:", np.std(arr_stats))

# Generación de datos
print("\nGeneración de datos:")
print("Arreglo de ceros:", np.zeros((2, 3)))
print("Arreglo de unos:", np.ones((3, 2)))
print("Arreglo de valores aleatorios:", np.random.rand(2, 2))

# Indexación y rebanado
print("\nIndexación y rebanado:")
print("Primer elemento de arr1d:", arr1d[0])
print("Última fila de arr2d:", arr2d[-1])
print("Elementos desde el segundo hasta el cuarto:", arr1d[1:4])

# Forma y tamaño de un arreglo
print("\nForma y tamaño de un arreglo:")
print("Forma de arr1d:", arr1d.shape)
print("Tamaño de arr2d:", arr2d.size)

# Cambiar la forma de un arreglo
print("\nCambiar la forma de un arreglo:")
print(arr2d.reshape(1, 9))

# Apilar y dividir arreglos
print("\nApilar y dividir arreglos:")
arr_stack = np.stack((arr1d, arr1d))
print("Apilamiento vertical:", np.vstack((arr1d, arr1d)))
print("Apilamiento horizontal:", np.hstack((arr1d, arr1d)))
print("División vertical:", np.vsplit(arr_stack, 2))
print("División horizontal:", np.hsplit(arr_stack, 2))
