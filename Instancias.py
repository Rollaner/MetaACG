# Cargador de instancias, junto con las soluciones para evaluar.
import os
import glob
from dataclasses import dataclass
from typing import List, Tuple, Union
import pandas as pd

@dataclass
class Instancia:
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


class DataLoader:
    def __init__(self, basePath: str = "Data/", basePathExt: str = "Data/EHOP_dataset/"):
        self.basePath = basePath
        self.basePathExt = basePathExt
        self.dataStore: dict[str, Instancia] = {}

    def cargarProblemas(self):
        problemas = ["graph_coloring", "traveling_salesman", "knapsack"]
        datasets = ["hard_dataset", "random_dataset"]
        for tipoProblema in problemas:
            for tipoDataset in datasets:
                CSV = os.path.join(self.basePathExt, tipoProblema, tipoDataset)
                self.procesarDatos(tipoProblema, tipoDataset, CSV)
        
        print(f"Cargadas {len(self.dataStore)} instancias de problemas.")

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
        archivosCSV = glob.glob(os.path.join(path, "*hard_sample.csv"))
        for csv in archivosCSV:
            print(csv)
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
                record = Instancia(
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

    def getDatosInstancia(self, claveInstancia: str) -> Instancia | None:
        return self.dataStore.get(claveInstancia)

    def getInstancia(self, claveInstancia: str) -> str | None:
        record = self.dataStore.get(claveInstancia)
        return record.problem if record else None
    
    def parsearSolucion(self, problemType: str, claveInstancia: str, solutionContent: str) -> Union[List[int], any, None]:
        if problemType == "graph_coloring":
            return self.parsearGC(claveInstancia, solutionContent)
        elif problemType == "traveling_salesman":
            return self.parsearTSP(claveInstancia, solutionContent)
        elif problemType == "knapsack":
            return self.parsearK(claveInstancia, solutionContent)
        else:
            print(f"Warning: No hay reglas de parseo de soluciones para: {problemType}")
            return None

    def getAllInstancias(self) -> List[Instancia]:
        return list(self.dataStore.values())
    
    def parsearTSP(self, claveInstancia: str, solutionContent: str) -> Tuple[int, List[int]] | None: #Pendiente, todavia no unterpreto bien los resultados
        return None

    def parsearGC(self, claveInstancia: str, solutionContent: str) -> Tuple[int, List[int]] | None:
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

    def parsearK(self, claveInstancia: str, solutionContent: str, Invertido=False) -> Tuple[int, List[int]] | None:
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

