from dataclasses import dataclass
import time
from typing import List, Tuple
import deprecated

@dataclass
class InstanciaPruebaK:
    length: int
    values: tuple[int,...]
    weights: tuple[int,...]
    solution: tuple[int,...]
    score: int
    capacity: int
    time: float

def cargarTest(dataTestStore, csv):
    with open(csv, 'r') as file: 
        while True:
            values,weights,selected = [], [], []
            key = file.readline().strip()
            if not key:
                break
            n = int(file.readline().split()[1])
            capacity = int(file.readline().split()[1])
            value = int(file.readline().split()[1])
            time = float(file.readline().split()[1])
            for i in range(n):
                stripped = file.readline().strip()
                if not stripped:
                    continue  
                parts = stripped.split(',')
                values.append(parts[1].strip('"'))
                weights.append(parts[2].strip('"'))
                selected.append(parts[3].strip())
            dataTestStore[key] = InstanciaPruebaK(
                length = n,
                values = tuple(values),
                weights = tuple(weights),
                solution = tuple(selected),
                score = value,
                capacity = capacity,
                time = time
            )
            eof = file.readline().strip()
            if eof == "-----":
                file.readline()
                continue
    return dataTestStore


def cargarKnapsackEHOP(data):
        lineas = [line.strip() for line in data.splitlines() if line.strip()]
        idx = 0
        n = int(lineas[idx])
        idx += 1
        values = []
        weights = []
        for _ in range(n):
            parts = lineas[idx].split()
            values.append(int(parts[1]))
            weights.append(int(parts[2]))
            idx += 1
        capacity = int(lineas[idx])
        idx += 1
        greedySolution, greedyScore,greedyTime = generarSolucionGreedyK(n,values,weights,capacity)
        solution, value, DPTime = generarSolucionDPK(n, values, weights, capacity)
        return solution, value, DPTime, n, sum(values)


def generarSolucion(claveInstancia: str, contenidoInstancia : str,subtipo: str):
    solucion, valor, tiempo, n, maxValue = cargarKnapsackEHOP(contenidoInstancia)
    if(subtipo == "inverted"):
        indexes = list(range(n))
        return list(set(indexes) - set(solucion)), maxValue - valor
    return solucion, valor

def generarSolucionGreedyK(nItems,values, weights, capacity):
    startTime = time.perf_counter()
    valuePerWeight = map(lambda x, y: x/y, values, weights)
    indexes = list(range(nItems))
    problem = sorted(zip(indexes,valuePerWeight), key = lambda x: x[1])
    soluciona = []
    solucionb = []
    weighta, weightb = 0, 0
    scorea, scoreb = 0, 0
    for i in range(nItems):
        element = problem.pop()
        newWeight = weights[element[0]] + weighta
        if newWeight > capacity:
            weightb = weights[element[0]] + weightb
            scoreb = scoreb + values[element[0]]
            solucionb.append(element[0])
            break;
        weighta = newWeight
        scorea = scorea + values[element[0]]
        soluciona.append(element[0])
    if scorea > scoreb: solucion = soluciona; score = scorea
    else: solucion = solucionb; score = scoreb
    endTime = time.perf_counter()
    TotalTime = endTime - startTime
    return tuple(solucion), score, TotalTime

def generarSolucionDPK(nItems,values, weights, capacity):
    startTime = time.perf_counter()  
    scores = [[-1 for _ in range(capacity+1)] for _ in range(nItems+1)]
    scores = exploracionRecursiva(scores,nItems,capacity,values,weights)
    solucion = []
    solucion = construirSolucion(nItems,capacity,scores,weights,solucion)
    endTime = time.perf_counter()
    TotalTime = endTime - startTime
    return tuple(solucion), scores[nItems][capacity], TotalTime

def exploracionRecursiva(valor, nItems, capacity, values, weights):
    if nItems == 0 or capacity <= 0:
        valor[nItems][capacity] = 0
        return valor
    if(valor[nItems-1][capacity] == -1):
        valor = exploracionRecursiva(valor,nItems-1,capacity,values,weights)
    if weights[nItems-1] > capacity:
        valor[nItems][capacity] = valor[nItems-1][capacity] 
    else:
        if(valor[nItems-1][capacity-weights[nItems-1]] == -1):
            valor = exploracionRecursiva(valor,nItems-1,capacity-weights[nItems-1],values,weights)
        valor[nItems][capacity] = max(valor[nItems-1][capacity], valor[nItems-1][capacity-weights[nItems-1]] + values[nItems-1])
    return valor
    
def construirSolucion(nItems,capacity,tablaValores, weights, solucion):
    if(nItems == 0): return 
    if(tablaValores[nItems][capacity] > tablaValores[nItems-1][capacity]):
        construirSolucion(nItems-1, capacity-weights[nItems-1],tablaValores,weights,solucion)
        solucion.append(nItems-1)
    else:
        construirSolucion(nItems-1, capacity, tablaValores,weights, solucion)
    return solucion


def parsearSolucion(claveInstancia: str, solutionContent: str) -> Tuple[int, List[int]] | None:
    stringValores = solutionContent.replace('\n', ' ').replace(',', ' ',).split()
    if not stringValores:
        return None
    try:
        valoresInt = [int(v) for v in stringValores]
        valorObtenido = valoresInt[0]
        arraySolucion = valoresInt[1:]
        return (valorObtenido, arraySolucion)   
    except ValueError as e:
        print(f"Error parseando la solucion de {claveInstancia}. El contenido era: '{solutionContent.strip()}'. Error: {e}")
    return None