from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import tsplib95 as tsp


NAME = "TSP"

@dataclass
class InstanciaPruebaTSP:
    name: str
    dimension: int
    tipoPesoAristas: str
    formatoPesoAristas: Optional[str] = None
    comment: Optional[str] = None
    pesoAristas: np.ndarray = field(default_factory=lambda: np.array([]))
    solution: tuple[int, ...] = field(default_factory=tuple)
    score: float = 0.0
    time: float = 0.0


def cargarTest(dataTestStore, tsp_file):
    with open(tsp_file, 'r') as f:
        key = tsp_file.split('/')[-1]
        name = None
        dimension = None
        tipoPesoAristas = None
        formatoPesoAristas = None
        comment = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(':')
            if len(parts) >= 2:
                nombre = parts[0].strip()
                datos = ':'.join(parts[1:]).strip()     
                if nombre == 'NAME':
                    name = datos
                elif nombre == 'DIMENSION':
                    dimension = int(datos)
                elif nombre == 'EDGE_WEIGHT_TYPE':
                    tipoPesoAristas = datos
                elif nombre == 'EDGE_WEIGHT_FORMAT':
                    formatoPesoAristas = datos
                elif nombre == 'COMMENT':
                    comment = datos
                elif nombre == 'EDGE_WEIGHT_SECTION':
                    edge_weights = leerPesosAristas(f, dimension, formatoPesoAristas)
                    break
        
        #solucion, puntaje, tiempo = generarSolucionGreedyTSP(edge_weights, dimension)
        
    #    dataTestStore[key] = InstanciaPruebaTSP(
    #       name=name,
    #       dimension=dimension,
    #       tipoPesoAristas=tipoPesoAristas,
    #       formatoPesoAristas=formatoPesoAristas,
    #       comment=comment,
    #       pesoAristas=edge_weights,
    #       solution=solucion,
    #       score=puntaje,
    #       time=tiempo
    #    )
    
    return dataTestStore

def leerPesosAristas(f, dimension, formatoPesoAristas):
    
    if formatoPesoAristas in ['UPPER_ROW''LOWER_DIAG_ROW']:
        valores = []
        for line in f:
            line = line.strip()
            if line == 'EOF' or not line:
                break
            valores.extend([int(x) for x in line.split()])
        matriz = np.zeros((dimension, dimension))
        #operador ternario. variable = valor1 if (condicion) else valor2. 
        indices = np.tril_indices(dimension)  if formatoPesoAristas == 'UPPER_ROW' else np.tril_indices(dimension) 
        matriz[indices] = valores
        matriz = matriz + matriz.T - np.diag(np.diag(matriz))
    
    elif formatoPesoAristas == 'FULL_MATRIX':
        for i in range(dimension):
            matriz = np.array([list(map(int, f.readline().split())) 
                   for _ in range(dimension)])

    return matriz

def parsearSolucion(claveInstancia: str, solutionContent: str) -> Tuple[int, List[int]] | None: #Pendiente, todavia no unterpreto bien los resultados
        stringValores = solutionContent.replace('\n', ' ').replace(',', ' ',).split()
        if not stringValores:
            return None
        try:
            soluciones = []
            i = 0
            while i < len(stringValores) - 1:
                cost = int(stringValores[i])
                if ',' in stringValores[i + 1]:
                    stringValores = stringValores[i + 1].replace(',', ' ').split()
                    tour = [int(v) for v in stringValores]
                    soluciones.append((cost, tour))
                    i += 2
                else:
                    i += 1
            if not soluciones:
                return None
            valorObtenido, arraySolucion = min(soluciones, key=lambda s: s[0])
            return (valorObtenido, arraySolucion)
        except ValueError as e:
            print(f"Error parseando la solucion de {claveInstancia}. El contenido era: '{solutionContent.strip()}'. Error: {e}")
        return None

def generarSolucion(claveInstancia: str,  contenidoInstancia : str,subtipo: str):
    return None

