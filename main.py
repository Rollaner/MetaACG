import ast
import pandas as pd
import matplotlib as plot
import numpy as np
import os
import PromptSampler
from dotenv import load_dotenv
import Generador
import Heuristicas.IteratedLocalSearch
import Heuristicas.SimmualtedAnnealing
import Heuristicas.TabuSearch
## Problemas: IDProblema, Descripcion / Componentes : ProblemaID, Representacion, Funcion Vencindad, Funcion ast.literal_evaluacion, Version
## Resultados: ProblemaID, componentVer, Metaheuristica, resultado
## Prompts: ProblemaID, promptBase, feedbackPrompt, feedBackComponents, feedbackResults, componentVer
def main():
    load_dotenv()
    pathDB= os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(pathDB, exist_ok=True)
    componentesPath = os.path.join(pathDB, 'componentes.csv')
    problemasPath = os.path.join(pathDB, 'problemas.csv')
    feedbackPath = os.path.join(pathDB,'feedback.csv')
    resultPath = os.path.join(pathDB,'resultados.csv')
    componenteDB = pd.read_csv(componentesPath)
    problemaDB = pd.read_csv(problemasPath)
    feedbackDB = pd.read_csv(feedbackPath)
    resultDB = pd.read_csv(resultPath)
    seed = 10
    problema = PromptSampler.sampleProblemaDB(problemaDB,seed)
    seedPrompt = PromptSampler.generateSeedPrompt(problema,componenteDB, seed)
    iterations = 10
    while iterations > 0:

        
        newPrompt = PromptSampler.updatePrompt(problema,componenteDB,resultDB, feedbackDB,seed)
        iterations = iterations - 1

    componenteDB.to_csv(componentesPath, index=False)
    problemaDB.to_csv(problemasPath,index=False)
    feedbackDB.to_csv(feedbackPath,index=False)
    resultDB.to_csv(resultPath,index=False)
    return 0


"Funcion para cargar componentes genericos, requiere sandboxing."
def cargarComponente(codigo: str):
    nombres = {}
    try:
        componente = ast.literal_eval(codigo,globals(),nombres)
        return componente
    except SyntaxError: 
        "Si el componente no es una expresion Lamda o representacion simple, se asume que es una funcion multilinea"
        exec(codigo,globals(),nombres)
        for key,value in nombres.items():
            if callable(value):
                print(f"Se ha cargado la funcion '{key}'")
                return value
        raise ValueError(f"No existen funciones encontradas en codigo cargado con exec()")
    except Exception as e:
        raise RuntimeError(f"no se ha podido cargar el codigo recuperado: {e}")

main()