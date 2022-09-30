import numpy as np
import pandas as pd

######### Funcion error #########
def funciones_penalizacion(genotipo,n):
    penalizacion_central = False
    central = genotipo[:n]
    penalizacion_desconexion = False
    n_indice = n
    indice_inicial = 0
    indice_final = n_indice
    matriz_indices = np.zeros([n,n])
    for i in range(n):
        contador = 0
        valor = genotipo[indice_inicial:indice_final]
        if i == 0:
            central = valor
            matriz_indices[n-1] = central
            if(sum(central) == 0):
                penalizacion_central = True
                break
        else:
            for j in valor:
                if j == 1:
                    matriz_indices[i-1][i-1] = 1
                    matriz_indices[i-1][i+contador] = 1
                contador += 1
        n_indice -= 1
        indice_inicial = indice_final
        indice_final = indice_final + n_indice
    busqueda_error = np.sum(matriz_indices, axis=0)
    suma_genotipo = np.sum(genotipo, axis=0)
    if suma_genotipo == 0:
        penalizacion_desconexion = True
    if 0 in busqueda_error:
       penalizacion_desconexion = True
    if not penalizacion_central and not penalizacion_desconexion:
        conexion_central = np.array([1 if x==1 else 0 for x in central])
        indices_conexion = np.where(conexion_central==1)[0]
        contador_unos = len(indices_conexion)
        contador = 0
        while contador <= contador_unos:
            for i in indices_conexion:
                fila = np.array(matriz_indices[i])
                columna = np.array(matriz_indices[:,i].T)
                indices_fila = np.where(fila==1)[0]
                indices_columna = np.where(columna[:-1]==1)[0]
                for j in indices_fila:
                    conexion_central[j]=1
                for k in indices_columna:
                    conexion_central[k]=1
            indices_conexion = np.where(conexion_central==1)[0]
            contador_unos = len(indices_conexion)
            contador += 1
        if conexion_central.sum() != n:
            penalizacion_desconexion = True
    return penalizacion_central or penalizacion_desconexion


### Metricas
def metricas(poblacion,distancias,valor_penalizacion, tamaño_poblacion):
    fitness = np.dot(poblacion,distancias)  ## Calcula funcion fitness
    fitness = fitness*valor_penalizacion
    fitness_generacion = fitness.sum()
    fitness_average = fitness/fitness_generacion
    suma_fitness_average = fitness_average.sum()
    maximo_fitness = 1-fitness_average
    ponderado = maximo_fitness*tamaño_poblacion
    tabla_promedio = pd.DataFrame({'fitness':fitness,'f/sum':fitness_average,'Max':maximo_fitness,'fi/f * tpobl':ponderado})
    #print(tabla_promedio.head())
    return ponderado,fitness.min(),np.argwhere(fitness==fitness.min())


#### Valores de cruce
def valores_de_cruce(tamaño, probabilidad, numero_conexiones):
    valor = int(tamaño*probabilidad)
    rng = np.random.default_rng()
    numbers = rng.choice(tamaño, size=valor*2, replace=False)
    punto_cruce = np.random.randint(1,numero_conexiones)
    vector = np.array(numbers).reshape(valor,2)
    return vector, punto_cruce

## Vector de penalzacion
def vector_penalzacion(poblacion_inicial,n):
    vector_penalizacion = []        
    for i in poblacion_inicial: ## verifica conexiones
        vector_penalizacion.append(funciones_penalizacion(i,n))
    penalizacion = [2000 if i else 1 for i in vector_penalizacion] ## Penaliza con un valor de 2000 si no se hacen todas las conexiones
    return penalizacion

#### Valores de mutacion
def valores_de_mutacion(tamaño, probabilidad, numero_conexiones):
    valor = int(tamaño*probabilidad*numero_conexiones)
    rng = np.random.default_rng()
    filas = rng.choice(numero_conexiones, size=valor, replace=False)
    columnas = rng.choice(tamaño, size=valor, replace=False)
    vector = [[x,y] for x,y in zip(columnas,filas)]
    return vector