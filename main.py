import ast
import argparse
import sys
import time
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
## Prompts: ProblemaID, promptBase, feedbackPrompt, feedBackComponents, feedbackResults, componentVe

def main():
    semilla = 1234
    #Inicializar datos
    load_dotenv()
    #Modificacion para pruebas, prepara modo batch por defecto. Estas lineas se tienen que eliminar cuando se empieze a optimizar
    if len(sys.argv) == 1:
        sys.argv.append('-b')
    pathDB= os.path.join(os.path.dirname(__file__), 'Data')
    #Fin modificacion para pruebas
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch',action='store_true', dest='batch_def', help='Definir problemas en batch, no optimiza')
    parser.add_argument('problem_ID', nargs='?',default=None,help="ID del problema a optimizar: Formato: Tipo_dataset_ID")
    args=parser.parse_args()
    print("Datos inicializados. Iniciando definicion matematica")
    #Problemas en batch
    if args.batch_def:
        definirBatch(instancias,llms)
    else:
    #Problema individual
        instancia:Instancia = instancias.getDatosInstancia("graph_coloring_random_dataset_in_house_9_8")
        respuesta = definirProblema(instancia)
        for i in 3:
            print(respuesta)
            feedback, resultados = evaluarDefinicion(instancia, respuesta)
            respuesta = refinarDefinicion(instancia,feedback, resultados)
    componenteDB.to_csv(componentesPath, index=False)
    problemaDB.to_csv(problemasPath,index=False)
    feedbackDB.to_csv(feedbackPath,index=False)
    resultDB.to_csv(resultPath,index=False)
    return 0

#Recalibración de las LLMS
def reiniciarLLMS():
    llms = generador()
    llms.cargarLLMs()
    return llms

def definirBatch(instancias:DataLoader,llms:generador):
    tiempoInicio = time.perf_counter()
    header = ['Instancia','Traje','Tipo de problema', 'Subtipo de problema', 'Iteracion', 'Respuesta', 'Feedback', 'Resultado esperado', 'tokens', 'tiempo']
    csvDefinicion = 'definicion_log.csv'
    i = 0
    try:
        with open(csvDefinicion, 'w', newline='', encoding='utf-8') as csvfile:
                escritor = csv.writer(csvfile)
                escritor.writerow(header)
        with open(csvDefinicion, 'a', newline='', encoding='utf-8') as csvfile:
            escritor = csv.writer(csvfile)
            for instancia in instancias.getAllInstancias():
                respuesta = definirProblema(llms,instancia)
                datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, i, respuesta.content[0]['text'], '', '', time.perf_counter()-tiempoInicio]
                escritor.writerow(datos)
                for j in range(3):
                    feedback, resultados = evaluarDefinicion(llms,instancia, respuesta.content[0]['text'])
                    respuesta = refinarDefinicion(llms,instancia,feedback, resultados)
                    datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, i, respuesta.content[0]['text'], feedback, resultados, time.perf_counter()-tiempoInicio]         
                    escritor.writerow(datos)
                reiniciarLLMS()
                i = i + 1
    except Exception as e:
                print(f"Error de escritura durante el proceso de definicion, no se han guardado los resultados {e}")
    print("Fin proceso de definicion matematica")



def definirProblema(llms,instancia:Instancia):
    ## Generar prompts con PromptSamplerDM
    prompt = PromptSamplerDM.generateSeedPrompt(instancia.problem)
    ## Prompt con tipo de problema. Este es menos exigente en las capacidades de razocinio de la LLM. Servira para pruebas a futuro
    #prompt = generateSeedPromptWithProblemTYpe(instanciaPrueba.problem,instanciaPrueba.problemType,semilla)
    ## Preparar enjambre de LLMs (Actualmente solo con ChatGPT)
    respuesta = llms.generarDefinicion(prompt)
    ## Falta guardar las respuestas
    return respuesta

def evaluarDefinicion(llms,instancia:Instancia, respuesta):
    valores = []
    for valor in respuesta.split(','):
        valores.append(valor.strip('"'))
    respuestaDef = valores [1]
    respuestaObj = valores[2]
    respuestaEval = valores[3]
    prompt =  PromptSamplerDM.generateFeedbackPromptNR(instancia.problem,respuestaDef,respuestaObj,respuestaEval,instancia.solutionValue)
    feedback = llms.generarFeedback(prompt)
    return feedback, instancia.solutionValue

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

#Utiliza las DB para guardar las respuestas con versionado. Eso significa que hay que acceder a los archivos y escribir sobre ellos. 
#Podemos pasar el problema definido en la fase 1 directamente, en vez de sacar uno de la DB. Eso nos deja con dos variantes: seed, y problema predefinido. En ambos casos resultDB, feedbackDB y componenteDB son necesarios
#actualmente solo esta evaluando 1 componente a la vez, tiene que evaluar sets de componentes. Representacion, Vecindad y Evaluacion
def optimizarProblemaAleatorio(problemaDB:pd.DataFrame,componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame,seed):
    problema = PromptSampler.sampleProblemaDB(problemaDB,seed)
    seedPrompt = PromptSampler.generateSeedPrompt(problema,componenteDB, seed)
    llms = generador()
    llms.cargarLLMs()
    respuestas = []
    iterations = 10
    respuestaInicial = llms.generarComponentes(seedPrompt)
    respuesta = respuestaInicial
    componenteDB.add(respuesta)
    while iterations > 0:
        resultado = evaluarComponentes(respuesta)
        resultDB.add(resultado)
        feedbackPrompt = PromptSampler.generateFeedbackPrompt(problema,respuesta,resultado)
        feedback = llms.generarFeedback(feedbackPrompt) 
        feedbackDB.add(feedback)
        newPrompt = PromptSampler.updatePrompt(problema,componenteDB,resultDB,feedback,seed)
        respuesta = llms.generarComponentes(newPrompt)
        respuestas.append(respuesta)
        componenteDB.add(respuesta)
        iterations = iterations - 1
        if iterations % 3 == 0:
            llms = reiniciarLLMS()

def optimizarProblemaPredefinido(problema,componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame,seed):
    seedPrompt = PromptSampler.generateSeedPrompt(problema,componenteDB, seed)
    llms = generador()
    llms.cargarLLMs()
    respuestas = []
    iterations = 10
    respuestaInicial = llms.generarComponentes(seedPrompt)
    respuesta = respuestaInicial
    componenteDB.add(respuesta.content[0]['text'])
    while iterations > 0:
        resultado = evaluarComponentes(respuesta.content[0]['text'])
        resultDB.add(resultado)
        feedbackPrompt = PromptSampler.generateFeedbackPrompt(problema,respuesta.content[0]['text'],resultado)
        feedback = llms.generarFeedback(feedbackPrompt) 
        feedbackDB.add(feedback)
        newPrompt = PromptSampler.updatePrompt(problema,componenteDB,resultDB,feedback,seed)
        respuesta = llms.generarComponentes(newPrompt)
        respuestas.append(respuesta.content[0]['text'])
        componenteDB.add(respuesta.content[0]['text'])
        iterations = iterations - 1
        if iterations % 3 == 0:
            llms = reiniciarLLMS()
    return respuestaInicial, respuestas

def evaluarComponentes(respuesta:str):
    return 0

#Funcion para cargar componentes genericos, requiere sandboxing.
def cargarComponente(codigo: str):
    nombres = {}
    try:
        componente = ast.literal_eval(codigo,globals(),nombres)
        return componente
    except SyntaxError: 
        #Si el componente no es una expresion Lamda o representacion simple, se asume que es una funcion multilinea
        exec(codigo,globals(),nombres)
        for key,value in nombres.items():
            if callable(value):
                print(f"Se ha cargado la funcion '{key}'")
                return value
        raise ValueError(f"No existen funciones encontradas en codigo cargado con exec()")
    except Exception as e:
        raise RuntimeError(f"no se ha podido cargar el codigo recuperado: {e}")

main()