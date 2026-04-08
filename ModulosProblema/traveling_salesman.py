from dataclasses import dataclass, field
import inspect
from typing import List, Optional, Tuple
import numpy as np
import tsplib95 as tsp


NAME = "TSP"

@dataclass
class InstanciaPruebaTSP:
    #name: str | None = None
    tipoPesoAristas: str = "none"
    formatoPesoAristas: Optional[str] = "none"
    pesoAristas: np.ndarray = field(default_factory=lambda: np.array([]))
    solution: tuple[int] = field(default_factory=tuple)
    dimension: int = 0
    #comment: Optional[str] 
    score: float = 0.0 
    time: float = 0.0 
   

def getSchema():
    return inspect.getsource(InstanciaPruebaTSP), InstanciaPruebaTSP, InstanciaPruebaTSP()

def _cargarDatos(datos): #Solo acepta tipo explicito. 
    with open(datos, 'r') as file:
        key = datos.split('/')[-1]
        #name = None
        dimension = None
        tipoPesoAristas = None
        formatoPesoAristas = None
        #comment = None
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(':')
            if len(parts) >= 2:
                nombreCampo = parts[0].strip()
                contCampo = ':'.join(parts[1:]).strip()     
                if nombreCampo == 'NAME':
                    _name = contCampo
                elif nombreCampo == 'DIMENSION':
                    dimension = int(contCampo)
                elif nombreCampo == 'EDGE_WEIGHT_TYPE':
                    tipoPesoAristas = contCampo
                elif nombreCampo == 'EDGE_WEIGHT_FORMAT':
                    formatoPesoAristas = contCampo
                elif nombreCampo == 'COMMENT':
                    _comment = contCampo
                elif nombreCampo == 'EDGE_WEIGHT_SECTION':
                    pesoAristas = _leerPesosAristas(file, dimension, formatoPesoAristas)
                    break
    return key, dimension, tipoPesoAristas, pesoAristas,formatoPesoAristas 

def _cargarDatosAlt(datos): #Solo acepta tipo explicito. 
    #name = None
    dimension = None
    tipoPesoAristas = None
    formatoPesoAristas = None
    #comment = None
    lines = datos.splitlines()
    for i, line in enumerate(lines):  
        line = line.strip()
        if not line:
            continue
        parts = line.split(':')
        if len(parts) >= 2:
            nombreCampo = parts[0].strip()
            contCampo = ':'.join(parts[1:]).strip()  
            if nombreCampo == 'NAME':
                _name = contCampo
            elif nombreCampo == 'DIMENSION':
                dimension = int(contCampo)
            elif nombreCampo == 'EDGE_WEIGHT_TYPE':
                tipoPesoAristas = contCampo
            elif nombreCampo == 'EDGE_WEIGHT_FORMAT':
                formatoPesoAristas = contCampo
            elif nombreCampo == 'COMMENT':
                _comment = contCampo
        elif line == 'EDGE_WEIGHT_SECTION':
            pesoAristas = _leerPesosAristas(iter(lines[i+1:]), dimension, formatoPesoAristas)
            break
    return dimension, tipoPesoAristas, pesoAristas,formatoPesoAristas 

def _leerPesosAristas(file, dimension, formatoPesoAristas):
    if formatoPesoAristas in ['UPPER_ROW','LOWER_DIAG_ROW']:
        valores = []
        for line in file:
            line = line.strip()
            if line == 'EOF' or not line:
                break
            valores.extend([int(x) for x in line.split()])
        matriz = np.zeros((dimension, dimension),dtype=np.float32)
        #operador ternario. variable = valor1 if (condicion) else valor2. 
        #triu es upper, tril es lower, k= indica si tiene la diagonal. 0 es si, 1 es no
        indices = np.triu_indices(dimension,k=1)  if formatoPesoAristas == 'UPPER_ROW' else np.tril_indices(dimension,k=0) 
        matriz[indices] = valores
        matriz = np.maximum(matriz, matriz.T)
    
    elif formatoPesoAristas == 'FULL_MATRIX':
        for i in range(dimension):
            matriz = np.array([list(map(int, file.readline().split())) 
                   for _ in range(dimension)])

    return matriz

def cargarTest(dataTestStore, filePath): #Se assume que los problemas son simetricos
    key, dimension, tipoPesoAristas, pesoAristas, formatoPesoAristas = _cargarDatos(filePath)
    solucion, puntaje, tiempo = _generarSolucionGreedyTSP(pesoAristas, dimension)
    dataTestStore[key] = InstanciaPruebaTSP(
#       name=name,
        dimension=dimension,
        tipoPesoAristas=tipoPesoAristas,
        formatoPesoAristas=formatoPesoAristas,
#       comment=comment,
        pesoAristas=pesoAristas,
        solution=solucion,
        score=puntaje,
        time=tiempo
    )
    return dataTestStore


def _generarSolucionGreedyTSP(pesoAristas: np.array ,dimension: int, subtipo="standard"):
    visitado = np.zeros(dimension, dtype=bool)
    ciudadActual = 0
    solucion = [ciudadActual]
    puntaje = 0
    tiempo = 0
    visitado[ciudadActual] = True

    for _ in range(dimension - 1):
        pesos = pesoAristas[ciudadActual].copy()
        pesos[visitado] = np.inf if subtipo != "inverted" else -np.inf

        siguienteDestino = np.argmin(pesos) if subtipo != "inverted" else np.argmax(pesos)
        
        puntaje += pesoAristas[ciudadActual, siguienteDestino]
        visitado[siguienteDestino] = True
        solucion.append(siguienteDestino)
        ciudadActual = siguienteDestino

    # Add return to start
    puntaje += pesoAristas[ciudadActual, solucion[0]]
    puntaje = int(puntaje)
    solucion = [int(c) + 1 for c in solucion] 
    return solucion, puntaje, tiempo

def parsearSolucion(claveInstancia: str, solutionContent: str):
    try:
        lines = [l.strip() for l in solutionContent.strip().splitlines() if l.strip()]
        soluciones = []
        i = 0
        while i < len(lines) - 1:
            try:
                cost = int(lines[i])
                tour = [int(v) for v in lines[i+1].replace(',', ' ').split()]
                soluciones.append((cost, tour))
                i += 2
            except ValueError:
                i += 1
        if not soluciones:
            return None
        valorObjetivo, arraySolucion = min(soluciones, key=lambda s: s[0])
        return valorObjetivo, arraySolucion
    except Exception as e:
        print(f"Error parseando solucion de {claveInstancia}: {e}")
        return None

def generarSolucion(claveInstancia: str,  contenidoInstancia,subtipo: str):
    dimension,_, pesoAristas, _,= _cargarDatosAlt(contenidoInstancia)
    solucion, valor, tiempo = _generarSolucionGreedyTSP(pesoAristas, dimension, subtipo)
    return solucion, valor

