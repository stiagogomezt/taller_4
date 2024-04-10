import numpy as np
#una forma de crear arreglos nd array es usando una lista 
miLista=[3,5,7,9,8]  
miArreglo= np.array(miLista) 
print (miArreglo.ndim)
print (miArreglo.shape)
miLista=[3,5,7,9,8,"hola"]  
miArreglo= np.array(miLista)

#se pueden crear matrices de arreglos en 2 dimesiones apartir listas 

miLista=[(1,3,5,7),(1,3,7,4.21)]
miArreglo=np.array(miLista, dtype=int)
print (miArreglo)

#usando funciones de relleno de arreglos
miArreglo=np.zeros((2,2))
print(miArreglo)
miArreglo= np.empty((6,3))
print(miArreglo)
miArreglo= np.arange(2,5)
print(miArreglo)
print(miArreglo.ndim)
miArreglo.reshape((2,3))
print(miArreglo.reshape)

