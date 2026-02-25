# Cargador de instancias, junto con las soluciones para evaluar.
import importlib
import os
import glob
from dataclasses import dataclass
from typing import List, Tuple, Union
import pandas as pd


@dataclass
class NLInstance:
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
    def __init__(self, basePath: str = "Data/", NLPath: str = "Data/NL/", testPath: str = "Data/Tests", modulePath: str = "ModulosProblema"):
        self.basePath = basePath
        self.NLPath = NLPath
        self.testPath = testPath
        self.modulePath = modulePath
        self.dataStore: dict[str, NLInstance] = {}
        self.dataTestStore: dict[str, any] = {}
        self._modulos: dict[str, object] = {}
        self.claves: dict[str,str] = {}
        

    def getClaves(self):
        return self.claves

    def cargarModulos(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        modulosDir = os.path.join(current_dir, self.modulePath)
        if not os.path.exists(modulosDir):
            os.makedirs(modulosDir)
            print(f"Directorio '{modulosDir}' creado.")
        archivos = [f[:-3] for f in os.listdir(modulosDir) 
                if f.endswith('.py') and not f.startswith('_')]
        return archivos

    def _getModulo(self,tipoProblema):
        if tipoProblema not in self._modulos:
            self._modulos[tipoProblema] = importlib.import_module(
                f"{self.modulePath}.{tipoProblema}"
            )
        return self._modulos[tipoProblema]

    def getSchema(self, tipoProblema):
        modulo = self._getModulo(self.getNombreCompleto(tipoProblema))
        return modulo.getSchema()

    def getNombreCompleto(self,tipoProblema):
        if tipoProblema in self.claves:
            nombreCompleto = self.claves[tipoProblema]
        elif tipoProblema in self.claves.values():
            nombreCompleto = tipoProblema
        else:
            print(f"Problema '{tipoProblema}' no reconocido. Disponibles: {list(self.claves.keys())}")
            return
        return  nombreCompleto

    def prepararClaves(self,archivos):
        claves = {}
        for archivo in archivos:
            nombreCompleto = archivo
            
            palabras = nombreCompleto.split('_')
            clave = ''.join(p[0].upper() for p in palabras)
            claves[clave] = nombreCompleto
        self.claves = claves
    
    def cargarProblemas(self,tipoProblema):
        listaModulos = self.cargarModulos()
        self.prepararClaves(listaModulos)
        datasets = ["hard_dataset", "random_dataset"]
        nombreCompleto = self.getNombreCompleto(tipoProblema)
        self.cargarPruebas(nombreCompleto),
        for tipoDataset in datasets:
            CSV = os.path.join(self.NLPath, nombreCompleto, tipoDataset)
            self.procesarDatos(nombreCompleto, tipoDataset, CSV)
        print(f"Cargadas {len(self.dataStore)} instancias de problemas.")

    def cargarPruebas(self, tipoProblema):
        testDir = os.path.join(self.testPath, tipoProblema)
        if not os.path.exists(testDir):
            print(f"No existen pruebas definidas para problemas de tipo {tipoProblema}")
            return
        try:
            archivos = [f for f in os.listdir(testDir) if os.path.isfile(os.path.join(testDir, f))]
            if not archivos:
                print(f"No se encontraron archivos de test en {testDir}")
                return
            path = os.path.join(testDir, archivos[0])
            modulo = self._getModulo(tipoProblema)
            self.dataTestStore = modulo.cargarTest(self.dataTestStore, path)
        except (ImportError, AttributeError) as e:
            print(f"Error cargando módulo {tipoProblema}: {e}")
        except Exception as e:
            print(f"Error inesperado cargando {tipoProblema}: {e}")


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
        extensionS = ".sol"
        for root, dirs, files in os.walk(nombre):
            if claveInstancia in dirs:
                dirInstancia = os.path.join(root, claveInstancia)
                break
        if dirInstancia:
            if tipoProblema == "graph_coloring":
                extensionI = ".col"
            elif tipoProblema == "traveling_salesman":
                 extensionI = ".tsp"
            else:
                 extensionI = ".in"
            archivoInstancia = os.path.join(dirInstancia, f"problem{extensionI}")
            archivoSolucion = os.path.join(dirInstancia, f"solution{extensionS}") if (subtipo == "standard") else os.path.join(dirInstancia, f"inverted_solution{extensionS}")
            try:
                with open(archivoInstancia, 'r') as pf:
                    instanceContent = pf.read()
                if os.path.exists(archivoSolucion):
                    with open(archivoSolucion, 'r') as sf: 
                        solutionContent = sf.read()
                    objectiveScore, parsedSolution = self.parsearSolucion(tipoProblema, dirInstancia, claveInstancia, solutionContent)
                else:
                    modulo = self._getModulo(tipoProblema)
                    parsedSolution, objectiveScore = modulo.generarSolucion(claveInstancia, instanceContent,subtipo) #generar solucion en base a los datos de instancia en texto plano
                    solutionContent = f"{objectiveScore}\n{", ".join(map(str, parsedSolution))}"
                    with open(archivoSolucion,"w") as f:
                        f.write(solutionContent)
                key=tipoProblema+'_'+tipoDataset+'_'+claveInstancia+'_'+traje+'_'+subtipo
                record = NLInstance(
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

    def getInstancias(self, claveInstancia: str) -> List[NLInstance] | None:
        return [
        instancia for key, instancia in self.dataStore.items() 
        if key.startswith(claveInstancia)
        ]

    def getInstancia(self, claveInstancia: str):
        return self.dataStore.get(claveInstancia)

    def getProblemaInstancia(self, claveInstancia: str) -> str | None:
        record = self.dataStore.get(claveInstancia)
        return record.problem if record else None
    
    def parsearSolucion(self, tipoProblema: str, dirInstancia: str, claveInstancia: str, solutionContent: str) -> Union[List[int], any, None]:
        try:
            archivos = [f for f in os.listdir(dirInstancia) if os.path.isfile(os.path.join(dirInstancia, f))]
            if not archivos:
                print(f"No se encontraron instancias en {dirInstancia}")
                return
            path = os.path.join(dirInstancia, archivos[0]) # lo podremos usar una vez que tengamos el sistema de inversos hecho
            modulo = self._getModulo(tipoProblema)
            objectiveScore, parsedSolution = modulo.parsearSolucion(claveInstancia,solutionContent)
        except (ImportError, AttributeError) as e:
            print(f"Error cargando módulo {tipoProblema}: {e}")
        except Exception as e:
            print(f"Error inesperado cargando {tipoProblema}: {e}")
        return objectiveScore, parsedSolution
    def getAllInstancias(self) -> List[NLInstance]:
        return list(self.dataStore.values())
    
    def getTestData(self)  -> List[any]:
        return list(self.dataTestStore.values())

    

