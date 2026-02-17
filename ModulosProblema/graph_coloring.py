from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class InstanciaPruebaGC:
    NoNodes: int
    Noedges: int
    solution: tuple[int,...]
    score: int
    time: float
    adj: Dict[int, Set[int]] = field(default_factory=dict)


def cargarTest(dataTestStore, dimacs, inverso=False):
        with open(dimacs, 'r') as f:
            key = dimacs.split('/')[-1]
            for line in f:
                parts = line.split()
                if not parts: continue
                if parts[0] == 'p':
                    NumNodes = int(parts[2])
                    NumEdges = int(parts[3])
                    ListaEdges = [set() for i in range(1, NumNodes + 1)]
                elif parts[0] == 'e':
                    nodo1, nodo2 = int(parts[1]), int(parts[2])
                    ListaEdges[nodo1-1].add(nodo2-1)
                    ListaEdges[nodo2-1].add(nodo1-1)
            solucion, puntaje, tiempo = generarSolucionGreedyGC(ListaEdges, NumNodes, inverso)
            dataTestStore[key] = InstanciaPruebaGC( NumNodes, NumEdges,solucion, puntaje, tiempo, ListaEdges)
        return dataTestStore

def generarSolucionGreedyGC(ListaEdges, NumNodes, inverso):
    #Ordernar por que tiene mas conexiones usando ListaEdges.sort().
    ListaEdgesDesc = sorted(ListaEdges, key= lambda x:  len(x), reverse = True)
    #Colorear con primer color disponible, marcar como usado. Normal: Nodos conectados no pueden compartir color. Inverso: Nodos no conectados no pueden compartir color
    #Repetir con cada nodo hasta que no queden mas nodos.
    solucion  = [-1] * NumNodes
    solucion[0] = 0

    if(inverso == True):

        disponibles = [False] * NumNodes
        for i in range(1,NumNodes):
            for j in ListaEdgesDesc[i]:
                if (solucion[j] != -1):
                    disponibles[solucion[j]-1] = True
            colorDisponible = 0;
            while colorDisponible < NumNodes:
                if (disponibles[colorDisponible]==True):
                    break
                colorDisponible += 1
            solucion[i] = colorDisponible
            for j in ListaEdgesDesc[i]:
                if (solucion[j] != -1):
                    disponibles[solucion[j]-1] = False

    else:

        disponibles = [True] * NumNodes
        for i in range(1,NumNodes):
            for j in ListaEdgesDesc[i]:
                if (solucion[j] != -1):
                    disponibles[solucion[j]-1] = False
            colorDisponible = 0;
            while colorDisponible < NumNodes:
                if (disponibles[colorDisponible]==True):
                    break
                colorDisponible += 1
            solucion[i] = colorDisponible
            for j in ListaEdgesDesc[i]:
                if (solucion[j] != -1):
                    disponibles[solucion[j]-1] = True
    puntaje = colorDisponible
    tiempo = 0
    return solucion, puntaje, tiempo

def generarSolucion(claveInstancia: str, contenidoInstancia: str, subtipo: str):
    #en este caso, debido a que inviertieron el grafo, la solucion inversa es equivalente a la normal
    #por lo que se copio la solucion original, por lo menos hasta que pueda probar bien el solver que esta aqui, no se usara
    return None


def parsearSolucion(claveInstancia: str, solutionContent: str) -> Tuple[int, List[int]] | None:
    stringValores = solutionContent.replace('\n', ' ').replace(',', ' ',).split()
    if not stringValores:
        return None
    try:
        valoresInt = [int(v) for v in stringValores]
        cantidadDeGrupos = valoresInt[0]
        arraySolucion = valoresInt[1:]
        return (cantidadDeGrupos, arraySolucion)   
    except ValueError as e:
        print(f"Error parseando la solucion de {claveInstancia}. El contenido era: '{solutionContent.strip()}'. Error: {e}")
    return None

