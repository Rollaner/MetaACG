from dataclasses import dataclass, field
import inspect
from typing import List, Optional, Tuple
import numpy as np
import tsplib95 as tsp


NAME = "TSP"

@dataclass
class InstanciaPruebaTSP:
    #name: str | None = None
    dimension: int = 0
    tipoPesoAristas: str | None = None
    formatoPesoAristas: Optional[str]
    #comment: Optional[str] 
    pesoAristas: np.ndarray = field(default_factory=lambda: np.array([]))
    solution: tuple[int] = field(default_factory=tuple)
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
    key = datos.split('/')[-1]
    #name = None
    dimension = None
    tipoPesoAristas = None
    formatoPesoAristas = None
    #comment = None
    for line in datos:
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
                pesoAristas = _leerPesosAristas(datos, dimension, formatoPesoAristas)
                break
    return key, dimension, tipoPesoAristas, pesoAristas,formatoPesoAristas 

def _leerPesosAristas(file, dimension, formatoPesoAristas):
    if formatoPesoAristas in ['UPPER_ROW''LOWER_DIAG_ROW']:
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
    solucion, puntaje, tiempo = generarSolucionGreedyTSP(pesoAristas, dimension)
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


def generarSolucionGreedyTSP(pesoAristas: np.array ,dimension: int, subtipo="standard"):
    solucion: list[int] = [] 
    puntaje:int = 0 
    tiempo = 0
    visitado = bytearray(dimension)
    ciudadesVisitadas: int = 0
    ciudadActual:int = 1 
    solucion.append(ciudadActual)
    while ciudadesVisitadas < dimension:
        if not visitado[ciudadActual]:
            posibilidades = np.where((np.frombuffer(visitado, dtype='u1') == 1 | pesoAristas[ciudadActual]  > 0), np.inf, pesoAristas[ciudadActual])
            if subtipo == "inverted":
                siguienteDestino = np.argmax(posibilidades)
            else:
                siguienteDestino = np.argmin(posibilidades)
            puntaje += pesoAristas[ciudadActual,siguienteDestino]
            visitado[ciudadActual] = 1
            ciudadesVisitadas -= 1
        ciudadActual = siguienteDestino
    return solucion, puntaje, tiempo

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

def generarSolucion(claveInstancia: str,  contenidoInstancia,subtipo: str):
    key, dimension, tipoPesoAristas, pesoAristas, formatoPesoAristas = _cargarDatosAlt(contenidoInstancia)
    solucion, valor, tiempo = generarSolucionGreedyTSP(pesoAristas, dimension, subtipo)
    return solucion, valor

