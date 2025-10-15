import ast
from dotenv import load_dotenv
import pandas as pd
import matplotlib as plot
import numpy as np
import os
import PromptSampler
import csv
import math
from typing import List, Union
from DefinicionMatematica import PromptSamplerDM
from DefinicionMatematica.Instancias import Instancia,DataLoader
from Generador import generador
import Heuristicas.IteratedLocalSearch
import Heuristicas.SimmualtedAnnealing
import Heuristicas.TabuSearch
## ProblemasOPT: IDProblema, Descripcion / Componentes : ProblemaID, Representacion, Funcion Vencindad, Funcion Evaluacion, Version
## Resultados: ProblemaID, componentVer, Metaheuristica, resultado
## Prompts: ProblemaID, promptBase, feedbackPrompt, feedBackComponents, feedbackResults, componentVer
def main():
    semilla = 1234
    #Inicializar datos
    load_dotenv()
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
    llms = generador()
    llms.cargarLLMs()
    print("Datos inicializados. Iniciando definicion matematica")
    #Problemas en batch
    datos = []
    header = ['Instancia', 'Iteracion', 'Respuesta', 'Feedback', 'Resultados']
    i = 0
    for instancia in instancias.getAllInstancias():
        respuesta = definirProblema(llms,instancia)
        print(respuesta)
        j = 0
        datos.append([instancia, j, respuesta, '', ''])
        #for i in range(3):
            #feedback, resultados = evaluarDefinicion(instancia, respuesta)
           # respuesta = refinarDefinicion(instancia,feedback, resultados)
          #  print(feedback,respuesta, resultados)
         #   datos.append([instancia, i, respuesta, feedback, resultados])
        #j = j +1
    #Problema individual
    #instanciaPrueba:Instancia = instancias.getDatosInstancia("graph_coloring_random_dataset_in_house_9_8")
    #respuesta = definirProblema(instanciaPrueba)
    #for i in 3:
    #    print(respuesta)
    #    feedback, resultados = evaluarDefinicion(instancia, respuesta)
    #    respuesta = refinarDefinicion(instancia,feedback, resultados)
    csvDefinicion = 'definicion_log.csv'
    try:
        with open(csvDefinicion, 'w', newline='', encoding='utf-8') as csvfile:
            escritor = csv.writer(csvfile)
            escritor.writerow(header)
            escritor.writerows(datos)
    except Exception as e:
        print("Error de escritura durante el proceso de definicion, no se han guardado los resultados")
    print("Fin proceso de definicion matematica")

    componenteDB.to_csv(componentesPath, index=False)
    problemaDB.to_csv(problemasPath,index=False)
    feedbackDB.to_csv(feedbackPath,index=False)
    resultDB.to_csv(resultPath,index=False)
    return 0

def reiniciarLLMS():
    return 0

def definirProblema(llms,instancia:Instancia):
    ## Generar prompts con PromptSamplerDM
    prompt = PromptSamplerDM.generateSeedPrompt(instancia.problem)
    ## Prompt con tipo de problema. Este es menos exigente en las capacidades de razocinio de la LLM. Servira para pruebas a futuro
    #prompt = generateSeedPromptWithProblemTYpe(instanciaPrueba.problem,instanciaPrueba.problemType,semilla)
    ## Preparar enjambre de LLMs (Actualmente solo con ChatGPT)
    respuesta = llms.generarDefinicion(prompt)
    ## Falta guardar las respuestas
    return respuesta
def evaluarDefinicion(llms,instancia:Instancia, respuesta:str):
    valores = []
    for valor in respuesta.split(','):
        valores.append(valor.strip('"'))
    respuestaDef = valores [1]
    respuestaObj = valores[2]
    respuestaEval = valores[3]
    MathResultados = probarDefinicion(respuestaObj,respuestaEval, instancia.parsedSolution)
    prompt =  PromptSamplerDM.generateFeedbackPrompt(instancia.problem,respuestaDef,respuestaObj,respuestaEval,MathResultados,instancia.solutionValue)
    feedback = llms.generarFeedback(prompt)
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
        objResult = eval(obj,safe_globals,local_vars) #Valor de la funcion objetivo
        evalResult = eval(eval, safe_globals, local_vars) #Valor de la funcion de evaluacion
        if isinstance(evalResult, (int, float)) and isinstance(objResult, (int, float)):
            return f"Evaluation Results: Objective Function Result: {objResult}, Evaluation Function Result: {evalResult}, Equal?: {objResult == evalResult}" 
        else: #Una funciuon objetivo y por extension la de evaluacion deben retornar valores numericos, no hacerlo seria un fallo catastrofico en la logica
            raise TypeError(f"Evaluation returned a non-numerical value: {type(objResult)} / {type(evalResult)}")

    except Exception as e: #Retornar excepciones si existen: Esto indica un fallo en la sintax de las funciones
        return f"Critical execution error for solution {parsedSolution}: {type(e).__name__} - {e}"

def refinarDefinicion(llms,instancia:Instancia,feedback:str, resultados):
    prompt = PromptSamplerDM.updatePrompt(instancia.problem, instancia.problemType,resultados,instancia.solutionValue,feedback)
    respuesta = llms.generarDefinicion(prompt)
    return respuesta

def optimizarProblema(problemaDB,componenteDB,resultDB,feedbackDB,seed):
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