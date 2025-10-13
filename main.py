import ast
import pandas as pd
import matplotlib as plot
import numpy as np
import os
import PromptSampler
import math
from typing import List, Union
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
    #Inicializar datos
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
    instancias = DataLoader()
    instancias.cargarDatos()
    print("Datos inicializados. Iniciando definicion matematica")
    respuesta, instancia = definirProblema(instancias)
    for i in 3:
        print(respuesta)
        feedback, resultados = evaluarDefinicion(instancia, respuesta)
        respuesta = refinarDefinicion(instancia,feedback, resultados)
    print("Fin proceso de definicion matematica")
    componenteDB.to_csv(componentesPath, index=False)
    problemaDB.to_csv(problemasPath,index=False)
    feedbackDB.to_csv(feedbackPath,index=False)
    resultDB.to_csv(resultPath,index=False)
    return 0

def reiniciarLLMS():
    return 0

def definirProblema(instancias:DataLoader):
    #TO-DO modificar promptsamplerDM para que pueda trabajar con los datos de instancia. Cargar LLMs, y ver como guardar las respuestas. 
    #La solucion para eso, se puede despues repetir para el evaluador. Puesto que el formato es un CSV, podemos extraer directamente lo que necesitamos.
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
    return respuesta, instanciaPrueba

def evaluarDefinicion(instancia:Instancia, respuesta:str):
    valores = []
    for valor in respuesta.split(','):
        valores.append(valor.strip('"'))
    respuestaDef = valores [1]
    respuestaObj = valores[2]
    respuestaEval = valores[3]
    MathResultados = probarDefinicion(respuestaObj,respuestaEval, instancia.parsedSolution)
    feedback =  PromptSamplerDM.generateFeedbackPrompt(instancia.problem,respuestaDef,respuestaObj,respuestaEval,MathResultados,instancia.solutionValue)
    return feedback, MathResultados

## Requiere revision y pruebas.
def probarDefinicion(obj: str, eval: str, parsedSolution: List[Union[int, float]]):
    # Definir entorno de ejecucion
    safe_globals = {
        '__builtins__': None,
        'abs': abs,
        'min': min,
        'max': max,
        'sum': sum,
        'math': math,
    }
    safe_globals.update(math.__dict__)
    #Pasando la solucion como variable local
    local_vars = {
        'solution': parsedSolution
    }
    #Evalucion
    try:
        objResult = eval(obj,safe_globals,local_vars)
        evalResult = eval(eval, safe_globals, local_vars)
        if isinstance(evalResult, (int, float)) and isinstance(objResult, (int, float)):
            return f"Evaluation Results: Objective Function Result: {objResult}, Evaluation Function Result: {evalResult}, Equal?: {objResult == evalResult}" 
        else: #Una funciuon objetivo y por extension la de evaluacion deben retornar valores numericos, no hacerlo seria un fallo catastrofico en la logica
            raise TypeError(f"Evaluation returned a non-numerical value: {type(objResult)} / {type(evalResult)}")

    except Exception as e: #Retornar excepciones si existen: Esto indica un fallo en la sintax de las funciones
        return f"Critical execution error for solution {parsedSolution}: {type(e).__name__} - {e}"


def refinarDefinicion(instancia:Instancia,feedback:str, resultados):
    return PromptSamplerDM.updatePrompt(instancia.problem, instancia.problemType,resultados,instancia.solutionValue,feedback)

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