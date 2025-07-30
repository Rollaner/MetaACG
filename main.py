import pandas as pd
import matplotlib as plot
import numpy as np
import os
import PromptSampler
import Generador
import Heuristicas.IteratedLocalSearch
import Heuristicas.SimmualtedAnnealing
import Heuristicas.TabuSearch

def main():
    pathDB= os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(pathDB, exist_ok=True)
    componentesPath = os.path.join(pathDB, 'componentes.csv')
    problemasPath = os.path.join(pathDB, 'problemas.csv')
    promptPath = os.path.join(pathDB,'prompts.csv')
    componenteDB = pd.read_csv(componentesPath)
    problemaDB = pd.read_csv(problemasPath)
    promptDB = pd.read_csv(promptPath)
    


    





    componenteDB.to_csv(componentesPath, index=False)
    problemaDB.to_csv(problemasPath,index=False)
    promptDB.to_csv(promptPath,index=False)
    return 0


"Funcion para cargar componentes genericos, requiere revision."
def cargarComponente(codigo: str):
    nombres = {}
    try:
        componente = eval(codigo,globals(),nombres)
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