import ast
import pandas as pd
import matplotlib as plot
import numpy as np
import os
import PromptSampler
from dotenv import load_dotenv
from DefinicionMatematica import Evaluador, PromptSamplerDM
from DefinicionMatematica.Instancias import Instancia,DataLoader
from Generador import generador
import Heuristicas.IteratedLocalSearch
import Heuristicas.SimmualtedAnnealing
import Heuristicas.TabuSearch
## Problemas: IDProblema, Descripcion / Componentes : ProblemaID, Representacion, Funcion Vencindad, Funcion ast.literal_evaluacion, Version
## Resultados: ProblemaID, componentVer, Metaheuristica, resultado
## Prompts: ProblemaID, promptBase, feedbackPrompt, feedBackComponents, feedbackResults, componentVer
def main():
    load_dotenv()
    semilla = 1234
    pathDB= os.path.join(os.path.dirname(__file__), 'Data')
    os.makedirs(pathDB, exist_ok=True)
    componentesPath = os.path.join(pathDB, 'componentes.csv')
    problemasPath = os.path.join(pathDB, 'problemas.csv')
    feedbackPath = os.path.join(pathDB,'feedback.csv')
    resultPath = os.path.join(pathDB,'resultados.csv')
    componenteDB = pd.read_csv(componentesPath)
    problemaDB = pd.read_csv(problemasPath)
    feedbackDB = pd.read_csv(feedbackPath)
    resultDB = pd.read_csv(resultPath)
    respuesta = definirProblema()
    print(respuesta)
    componenteDB.to_csv(componentesPath, index=False)
    problemaDB.to_csv(problemasPath,index=False)
    feedbackDB.to_csv(feedbackPath,index=False)
    resultDB.to_csv(resultPath,index=False)
    return 0

def reiniciarLLMS():
    return 0

def definirProblema():
    #TO-DO modificar promptsamplerDM para que pueda trabajar con los datos de instancia. Cargar LLMs, y ver como guardar las respuestas. 
    #La solucion para eso, se puede despues repetir para el evaluador. Puesto que el formato es un CSV, podemos extraer directamente lo que necesitamos.
    instancias = DataLoader()
    instancias.cargarDatos()
    instanciaPrueba:Instancia = instancias.getDatosInstancia("graph_coloring_random_dataset_in_house_9_8")
    ## Generar prompts con PromptSamplerDM
    prompt = PromptSamplerDM.generateSeedPrompt(instanciaPrueba.problem)
    respuesta = prompt
    ## Prompt con tipo de problema. Este es menos exigente en las capacidades de razocinio de la LLM. Servira para pruebas a futuro
    #prompt = generateSeedPromptWithProblemTYpe(instanciaPrueba.problem,instanciaPrueba.problemType,semilla)
    ## Preparar enjambre de LLMs (Actualmente solo con ChatGPT)
    #llms=generador()
    #llms.cargarLLMs
    #respuesta = llms.generarDefinicion(prompt)
    ## Falta guardar las respuestas
    return respuesta


def optimizarProblema(problemaDB,componenteDB,resultDB,feedbacDB,seed):
    problema = PromptSampler.sampleProblemaDB(problemaDB,seed)
    seedPrompt = PromptSampler.generateSeedPrompt(problema,componenteDB, seed)
    iterations = 10
    while iterations > 0:
        newPrompt = PromptSampler.updatePrompt(problema,componenteDB,resultDB, feedbackDB,seed)
        iterations = iterations - 1



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