# Cargador de instancias, junto con las soluciones para evaluar.
import os
import glob
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Dict, Set
import pandas as pd

@dataclass
class InstanciaEHOP:
    problemType: str #Tipo del problema: TSP, GC, o Knapsack
    problemCostume: str #Variante, o "traje" del problema
    problemSubType: str #Standard o invertido
    datasetType: str #Tipo del dataset del problema (Hard o Random)
    claveInstancia: str  # ID de la instancia: "in_house_6_0"
    problem: str #Problema en formate de lenguaje natural
    instanceContent: str #Valores numericos de la instancia como tal: AKA, los valores de EHOP en bruto
    solutionContent: str #Valores de la solucion optima
    parsedSolution: Union[List[int], List[float], List[List[int]], None] # Placeholder, solucion en fomato utilizable por codigo
    objectiveScore: int #Valor esperado de la funcion objetivo

@dataclass
class InstanciaPruebaK:
    length: int
    values: tuple[int,...]
    weights: tuple[int,...]
    solution: tuple[int,...]
    score: int
    scoreWeight: int
    time: float

@dataclass
class InstanciaPruebaGC:
    NoNodes: int
    Noedges: int
    solution: tuple[int,...]
    score: int
    time: float
    adj: Dict[int, Set[int]] = field(default_factory=dict)


class DataLoader:

    

    def __init__(self, basePath: str = "Data/", basePathExt: str = "Data/EHOP_dataset/"):
        self.basePath = basePath
        self.basePathExt = basePathExt
        self.dataStore: dict[str, InstanciaEHOP] = {}
        self.dataTestStore: dict[str, any] = {}

    def cargarProblemas(self,tipoProblema):
        problemas = ["graph_coloring", "traveling_salesman", "knapsack"]
        datasets = ["hard_dataset", "random_dataset"]
        if tipoProblema == "K":
           self.cargarPruebas("knapsack")
        if tipoProblema == "GC":
           self.cargarPruebas("graph_coloring")
        for tipoProblema in problemas:
            for tipoDataset in datasets:
                CSV = os.path.join(self.basePathExt, tipoProblema, tipoDataset)
                self.procesarDatos(tipoProblema, tipoDataset, CSV)
        
        print(f"Cargadas {len(self.dataStore)} instancias de problemas.")

    def cargarPruebas(self, tipoProblema):
        problemas = ["graph_coloring", "traveling_salesman", "knapsack"]
        if tipoProblema not in problemas:
            print(f"No existen pruebas definidas para problemas de tipo " + tipoProblema)
            return
        if tipoProblema ==  "knapsack":
            testPath = os.path.join(self.basePath, "Psinger_dataset/knapPI_11_20_1000.csv") #partamos con uno chico de momento
            self.cargarTestKnapsack(testPath)

    def cargarComponentes(self):
        problemas = ["graph_coloring", "traveling_salesman", "knapsack"]
        datasets = ["hard_dataset", "random_dataset"]
        for tipoProblema in problemas:
            for tipoDataset in datasets:
                JSONL = os.path.join(self.basePath, tipoProblema, tipoDataset)
                self.procesarComponentes(tipoProblema, tipoDataset, JSONL)
        print(f"Cargadas {len(self.dataStore)} instancias de problemas.")

    def procesarComponentes(self, path: str):
        archivoJSONL = glob.glob(os.path.join(path, "problemas.jsonl"))
        problemaDB = pd.read_json(archivoJSONL,lines=True)
        return problemaDB[::5]

    def procesarDatos(self, tipoProblema: str, tipoDataset: str, path: str):
        archivosCSV = glob.glob(os.path.join(path, "*sample.csv"))
        for csv in archivosCSV:
            with open(csv, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue  
                    try:
                        parts = line.split(',',3)
                        if len(parts) != 4:
                            print(f"Saltando linea malformada en {csv}: {line}")
                            continue
                        claveInstancia = parts[0].strip('"')
                        traje = parts[1].strip('"')
                        subtipo = parts[2].strip('"')
                        rawData = parts[3].strip()
                        if rawData.startswith('"') and rawData.endswith('"'):
                            innerContent = rawData[1:-1]
                            if innerContent.startswith("('") and  innerContent.endswith("')"):
                                problem = innerContent[2:-2]
                            else:
                                problem  = innerContent
                                print(f"Warning: Prefijo inesperado en {claveInstancia}")
                        else:
                            problem = rawData  
                        self.associarDatos(tipoProblema, tipoDataset, traje, subtipo, path, claveInstancia, problem)
                    except Exception as e:
                        print(f"Error procesando la linea en {csv}: {line} -> {e}")

    def associarDatos(self, tipoProblema: str, tipoDataset: str, traje:str, subtipo:str, nombre: str, claveInstancia: str, problem: str):
        dirInstancia = None
        for root, dirs, files in os.walk(nombre):
            if claveInstancia in dirs:
                dirInstancia = os.path.join(root, claveInstancia)
                break
        if dirInstancia:
            if tipoProblema == "graph_coloring":
                extensionI, extensionS = ".col", ".sol"
            elif tipoProblema == "traveling_salesman":
                 extensionI, extensionS = ".tsp", ".sol" 
            else:
                 extensionI, extensionS = ".in", ".sol" 
            archivoInnstancia = os.path.join(dirInstancia, f"problem{extensionI}")
            archivoSolucion = os.path.join(dirInstancia, f"solution{extensionS}")
            try:
                with open(archivoInnstancia, 'r') as pf:
                    instanceContent = pf.read()
                with open(archivoSolucion, 'r') as sf:
                    solutionContent = sf.read()
                objectiveScore, parsedSolution = self.parsearSolucion(tipoProblema, claveInstancia, solutionContent)
                key=tipoProblema+'_'+tipoDataset+'_'+claveInstancia+'_'+traje+'_'+subtipo
                record = InstanciaEHOP(
                    problemType=tipoProblema, #Tipo del problema: TSP, GC, o Knapsack
                    datasetType=tipoDataset, #Tipo del dataset del problema (Hard o Random)
                    problemCostume = traje, #Variante, o "traje" del problema
                    problemSubType = subtipo, #Standard o invertido
                    claveInstancia=key, # ID de la instancia: "in_house_6_0_hard_dataset_graph_coloring"
                    problem=problem, #Problema en formate de lenguaje natural
                    instanceContent=instanceContent, #Valores numericos de la instancia como tal: AKA, los valores de EHOP en bruto
                    solutionContent=solutionContent, #Valores de la solucion optima
                    parsedSolution=parsedSolution, # Placeholder, solucion en fomato utilizable por codigo
                    objectiveScore=objectiveScore  #Valor esperado de la funcion objetivo
                )
                self.dataStore[key] = record
            except FileNotFoundError:
                print(f"Archivo de valores del problema o solucion no encontrados para: {key} in {dirInstancia}")
            except Exception as e:
                print(f"Error cargando archivos de {key}: {e}")
                return
        else:
            print(f"No se encontro el directorio para: {key} in {nombre}")

    def getInstancias(self, claveInstancia: str) -> List[InstanciaEHOP] | None:
        return [
        instancia for key, instancia in self.dataStore.items() 
        if key.startswith(claveInstancia)
        ]

    def getProblemaInstancia(self, claveInstancia: str) -> str | None:
        record = self.dataStore.get(claveInstancia)
        return record.problem if record else None
    
    def parsearSolucion(self, problemType: str, claveInstancia: str, solutionContent: str) -> Union[List[int], any, None]:
        if problemType == "graph_coloring":
            return self.parsearGCEHOP(claveInstancia, solutionContent)
        elif problemType == "traveling_salesman":
            return self.parsearTSPEHOP(claveInstancia, solutionContent)
        elif problemType == "knapsack":
            return self.parsearKEHOP(claveInstancia, solutionContent)
        else:
            print(f"Warning: No hay reglas de parseo de soluciones para: {problemType}")
            return None

    def getAllInstancias(self) -> List[InstanciaEHOP]:
        return list(self.dataStore.values())
    
    def getTestData(self)  -> List[any]:
        return list(self.dataTestStore.values())

    def parsearTSPEHOP(self, claveInstancia: str, solutionContent: str) -> Tuple[int, List[int]] | None: #Pendiente, todavia no unterpreto bien los resultados
        return None

    def parsearGCEHOP(self, claveInstancia: str, solutionContent: str) -> Tuple[int, List[int]] | None:
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

    def parsearKEHOP(self, claveInstancia: str, solutionContent: str, Invertido=False) -> Tuple[int, List[int]] | None:
        stringValores = solutionContent.replace('\n', ' ').replace(',', ' ',).split()
        if not stringValores:
            return None
        if Invertido:
            return None
        try:
            valoresInt = [int(v) for v in stringValores]
            cantidadDeGrupos = valoresInt[0]
            arraySolucion = valoresInt[1:]
            return (cantidadDeGrupos, arraySolucion)   
        except ValueError as e:
            print(f"Error parseando la solucion de {claveInstancia}. El contenido era: '{solutionContent.strip()}'. Error: {e}")
        return None

    def cargarTestKnapsack(self, csv):
        with open(csv, 'r') as file: 
            while True:
                values,weights,selected = [], [], []
                key = file.readline().strip()
                if not key:
                    break
                n = int(file.readline().split()[1])
                cost = int(file.readline().split()[1])
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
                self.dataTestStore[key] = InstanciaPruebaK(
                    length = n,
                    values = tuple(values),
                    weights = tuple(weights),
                    solution = tuple(selected),
                    score = value,
                    scoreWeight = cost,
                    time = time
                )
                eof = file.readline().strip()
                if eof == "-----":
                    file.readline()
                    continue

    def cargarTestGC(self, dimacs):
        with open(dimacs, 'r') as f:
            key = dimacs.split('/')[-1]
            ListaEdges = {}
            NoNodes = 0
            NoEdges = 0
            for line in f:
                parts = line.split()
                if not parts: continue
                if parts[0] == 'p':
                    NoNodes = int(parts[2])
                    NoEdges = int(parts[3])
                    ListaEdges = {i: set() for i in range(1, NoNodes + 1)}
                elif parts[0] == 'e':
                    nodo1, nodo2 = int(parts[1]), int(parts[2])
                    # Graphs in DIMACS are usually undirected
                    ListaEdges[nodo1].add(nodo2)
                    ListaEdges[nodo2].add(nodo1)
            solucion, puntaje, tiempo = generarSolucionGC(ListaEdges)
            self.dataTestStore[key] = InstanciaPruebaGC( NoNodes, NoEdges,solucion, puntaje, tiempo, ListaEdges)