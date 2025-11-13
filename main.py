import ast
import argparse
import sys
import time
from dotenv import load_dotenv
import pandas as pd
import matplotlib as plot
import numpy as np
import os
import Optimizacion.PromptSamplerOP as PromptSamplerOP
import csv
import math
import json
import random
import itertools
from typing import List, Union,Callable, Any, Dict
from DefinicionMatematica import DefinicionMat
from Instancias import DataLoader
from Generador import generador
import Heuristicas.IteratedLocalSearch
import Heuristicas.SimmulatedAnnealing
import Heuristicas.TabooSearch
## ProblemasOPT: IDProblema, Descripcion / Componentes : ProblemaID, Representacion, Funcion Vencindad, Funcion Evaluacion, Version
## Resultados: ProblemaID, componentVer, Metaheuristica, resultado
## Prompts: ProblemaID, promptBase, feedbackPrompt, feedBackComponents, feedbackResults, componentVe

def main():
    semilla = 1234
    iteraciones = 9
    #Inicializar datos
    load_dotenv()
    #Modificacion para pruebas, prepara modo batch por defecto. Estas lineas se tienen que eliminar cuando se empieze a optimizar
    if len(sys.argv) == 1:
        sys.argv.append('-o')
    pathDB= os.path.join(os.path.dirname(__file__), 'Data')
    #Fin modificacion para pruebas
    os.makedirs(pathDB, exist_ok=True)
    componentesPath = os.path.join(pathDB, 'componentes.jsonl')
    problemasPath = os.path.join(pathDB, 'problemas.jsonl')
    feedbackPath = os.path.join(pathDB,'feedback.jsonl')
    resultPath = os.path.join(pathDB,'resultados.jsonl')
    instancias = DataLoader()
    instancias.cargarProblemas()
    llms = generador()
    llms.cargarLLMs()
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch',action='store_true', dest='batch_def', help='Definir problemas en batch, no optimiza')
    parser.add_argument('-o', '--opt',action='store_true', dest='skip_def', help='Solo optimizar, no define')
    parser.add_argument('problem_ID', nargs='?',default=None,help="ID del problema a optimizar: Formato: Tipo_dataset_ID")
    args=parser.parse_args()
    if args.skip_def:
        problemaDB = pd.read_json(problemasPath,lines=True)
        if os.path.exists(componentesPath) and os.path.exists(resultPath) and os.path.exists(feedbackPath):
            componenteDB = pd.read_json(componentesPath,lines=True)
            feedbackDB = pd.read_json(feedbackPath,lines=True)
            resultDB = pd.read_json(resultPath,lines=True)
        else:
            componenteDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Evaluacion', 'Vecindad', 'Perturbacion','Version'])
            feedbackDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Componente','Version', 'Feedback'])
            resultDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Evaluacion', 'Vecindad', 'Perturbacion', 'Resultados','Solucion','Valor Optimo', 'Metaheuristica', 'Tiempo'])
        for tuplaProblema in problemaDB.itertuples(index=False):    
            componenteDB, feedbackDB, resultDB = optimizarProblemaPredefinido(tuplaProblema, componenteDB,resultDB,feedbackDB, iteraciones)
            componenteDB.to_json(componentesPath,orient='records',lines=True)
            feedbackDB.to_json(feedbackPath,orient='records',lines=True)
            resultDB.to_json(resultPath,orient='records',lines=True)
        return 0
    #Problemas en batch
    if args.batch_def:
        print("Datos inicializados. Iniciando definicion matematica")
        DefinicionMat.definirBatch(instancias,llms, problemasPath, tipo="graph_coloring")
        return 0
    else:
    #Problema individual
        respuesta,feedback,resultados = DefinicionMat.definirIndividual(instancias,llms,problemasPath,"graph_coloring_random_dataset_in_house_9_8")
    #componenteDB = pd.read_json(componentesPath,lines=True)
    problemaDB = pd.read_json(problemasPath,lines=True)
    #feedbackDB = pd.read_json(feedbackPath,lines=True)
    #resultDB = pd.read_json(resultPath,lines=True)
    #componenteDB.to_json(componentesPath,lines=True)
    #feedbackDB.to_json(feedbackPath,lines=True)
    #resultDB.to_json(resultPath,lines=True)
    return 0

#Utiliza las DB para guardar las respuestas con versionado. Eso significa que hay que acceder a los archivos y escribir sobre ellos. 
#Podemos pasar el problema definido en la fase 1 directamente, en vez de sacar uno de la DB. Eso nos deja con dos variantes: seed, y problema predefinido. En ambos casos resultDB, feedbackDB y componenteDB son necesarios
#actualmente solo esta evaluando 1 componente a la vez, tiene que evaluar sets de componentes. Representacion, Vecindad y Evaluacion
def optimizarProblemaAleatorio(problemaDB:pd.DataFrame,componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame,semilla, iteraciones):
        rawProblema = PromptSamplerOP.sampleProblemaDB(problemaDB,semilla)
        componenteDB, feedbackDB, resultDB = optimizarProblemaPredefinido(rawProblema, componenteDB, resultDB, feedbackDB, iteraciones)
        return componenteDB, feedbackDB, resultDB

def optimizarProblemaPredefinido(problema,componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame, iteraciones):
        problemaID, defProblema, solucion, objetivo, seedPrompt = PromptSamplerOP.generateSeedPrompt(problema)
        print(problemaID, solucion, objetivo)
        llms = generador()
        llms.cargarLLMs()
        respuestaInicial = llms.generarComponentes(seedPrompt)
        textoBruto = respuestaInicial.content[0]['text']
        componentes = json.loads(textoBruto)
        print(componentes)
        respuestas = [] #Temporal, para gguardar todas las respuestas de generacion de componentes para tener la metadata.
        feedbacks = []
        i = 0
        while i < iteraciones:
            resultadoSA, tiempoSA = cronometrarFuncion(evaluarComponentesSA ,componentes)
            resultadoILS, tiempoILS = cronometrarFuncion(evaluarComponentesILS, componentes)
            resultadoTS, tiempoTS = cronometrarFuncion(evaluarComponentesTS, componentes)
            print(resultadoSA)
            print(resultadoILS)
            print(resultadoTS)
            resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoSA,solucion, objetivo, "SA", tiempoSA)
            resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoILS,solucion, objetivo, "ILS", tiempoILS)
            resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoTS,solucion, objetivo, "TS", tiempoTS)
            feedbackPrompt = PromptSamplerOP.generateFeedbackPrompt(defProblema,componentes,resultadoSA, resultadoILS,resultadoTS, solucion, objetivo) #Necesita trabajar con el nuevo sistema de JSON
            feedback = llms.generarFeedback(feedbackPrompt) 
            feedbacks.append(feedback)
            feedbackTexto = feedback.content[0]['text']
            print(feedbackTexto)
            feedbackDB = guardarFeedback(problemaID,feedbackDB,componentes['REPRESENTATION'], componentes, i,feedbackTexto)
            newPrompt = PromptSamplerOP.updatePromptOS(defProblema,componentes,resultadoSA,feedbackTexto)
            print(newPrompt)
            respuesta = llms.generarComponentes(newPrompt)
            respuestas.append(respuesta)
            textoBruto = respuesta.content[0]['text']
            print(textoBruto)
            componentes = json.loads(textoBruto)
            componenteDB = guardarComponentes(problemaID,componenteDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'], i)
            i = i + 1
            if i % 3 == 0:
                llms = DefinicionMat.reiniciarLLMSDef()
                print("Maquinas re-instanciadas")
        resultadoSA, tiempoSA = cronometrarFuncion(evaluarComponentesSA ,componentes)
        resultadoILS, tiempoILS = cronometrarFuncion(evaluarComponentesILS, componentes)
        resultadoTS, tiempoTS = cronometrarFuncion(evaluarComponentesTS, componentes)
        print(resultadoSA)
        print(resultadoILS)
        print(resultadoTS)
        resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoSA,solucion, objetivo, "SA", tiempoSA)
        resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoILS,solucion, objetivo, "ILS", tiempoILS)
        resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoTS,solucion, objetivo, "TS", tiempoTS)        
        return componenteDB, feedbackDB, resultDB

def evaluarComponentesSA(componentes):
    #Guardar excepciones como string, para retornarlas como debug
    try:
        solucionPrueba = ast.literal_eval(componentes['SAMPLE_SOL'].strip().replace(' ', ''))
    except Exception as e:
        return(f"Failed to load SAMPLE_SOL: {e}")
    try:
        evaluacion = cargarComponente(componentes['EVAL_CODE'])
        print(f"Loaded 'EVAL_CODE' into variable 'evaluate_solution'. Name: {eval.__name__}")
    except Exception as e:
        return(f"Failed to load EVAL_CODE: {e}")
    try:
        vecindad = cargarComponente(componentes['NB_CODE'])
        print(f"Loaded 'NB_CODE' into variable 'generate_neighbour'. Name: {vecindad.__name__}")
    except Exception as e:
        return(f"Failed to load NB_CODE: {e}")
    try:
        valor = evaluacion(solucionPrueba)
    except Exception as e:
        return(f"Failed to evaluate SAMPLE_SOL with EVAL_CODE: {e}")
    try:
        resultadoSA = Heuristicas.SimmulatedAnnealing.SA(solucionPrueba,solucionPrueba,valor,vecindad,evaluacion,1000,10,0.9)
    except Exception as e:
        return(f"Failed to run target heuristic: {e}.  Signature def SA(solution,best_sol, best_score, generate_neighbour(), evaluate_solution(), TEMP, MIN_TEMP, cooling_factor)")
    #Cargar heuristicas, retornar resultados de cada una.
    return resultadoSA

def evaluarComponentesILS(componentes):
    #Guardar excepciones como string, para retornarlas como debug
    try:
        solucionPrueba = ast.literal_eval(componentes['SAMPLE_SOL'].strip().replace(' ', ''))
    except Exception as e:
        return(f"Failed to load SAMPLE_SOL: {e}")
    try:
        evaluacion = cargarComponente(componentes['EVAL_CODE'])
        print(f"Loaded 'EVAL_CODE' into variable 'evaluate_solution'. Name: {eval.__name__}")
    except Exception as e:
        return(f"Failed to load EVAL_CODE: {e}")
    try:
        vecindad = cargarComponente(componentes['NB_CODE'])
        print(f"Loaded 'NB_CODE' into variable 'generate_neighbour'. Name: {vecindad.__name__}")
    except Exception as e:
        return(f"Failed to load NB_CODE: {e}")
    try:
        perturb = cargarComponente(componentes['PERTURB_CODE'])
        print(f"Loaded 'PERTURB_CODE' into variable 'perturb_solution'. Name: {perturb.__name__}")
    except Exception as e:
        return(f"Failed to load PERTURB_CODE: {e}")
    try:
        valor = evaluacion(solucionPrueba)
    except Exception as e:
        return(f"Failed to evaluate SAMPLE_SOL with EVAL_CODE: {e}")
    try:
        resultadoSA = Heuristicas.IteratedLocalSearch.ILS(solucionPrueba,solucionPrueba,valor,vecindad,perturb,evaluacion,44,0.1) ##44 iteraciones es lo mismo que tiene el de SA, en base a los valores de 1000, 10 y 0,9
    except Exception as e:
        return(f"Failed to run target heuristic: {e}.  Signature def ILS(solution,best_sol, best_score, generate_neighbour(),perturb_solution(), evaluate_solution(), iterations, aceptance_rate)")
    #Cargar heuristicas, retornar resultados de cada una.
    return resultadoSA

def evaluarComponentesTS(componentes):
    #Guardar excepciones como string, para retornarlas como debug
    try:
        solucionPrueba = ast.literal_eval(componentes['SAMPLE_SOL'].strip().replace(' ', ''))
    except Exception as e:
        return(f"Failed to load SAMPLE_SOL: {e}")
    try:
        evaluacion = cargarComponente(componentes['EVAL_CODE'])
        print(f"Loaded 'EVAL_CODE' into variable 'evaluate_solution'. Name: {eval.__name__}")
    except Exception as e:
        return(f"Failed to load EVAL_CODE: {e}")
    try:
        vecindad = cargarComponente(componentes['NB_CODE'])
        print(f"Loaded 'NB_CODE' into variable 'generate_neighbour'. Name: {vecindad.__name__}")
    except Exception as e:
        return(f"Failed to load NB_CODE: {e}")
    try:
        perturb = cargarComponente(componentes['PERTURB_CODE'])
        print(f"Loaded 'PERTURB_CODE' into variable 'perturb_solution'. Name: {perturb.__name__}")
    except Exception as e:
        return(f"Failed to load PERTURB_CODE: {e}")
    try:
        valor = evaluacion(solucionPrueba)
    except Exception as e:
        return(f"Failed to evaluate SAMPLE_SOL with EVAL_CODE: {e}")
    try:
        resultadoTS = Heuristicas.TabooSearch.TS(solucionPrueba,solucionPrueba,valor,vecindad,evaluacion,44,10,7)
    except Exception as e:
        return(f"Failed to run target heuristic: {e}.  Signature def TS(solution,best_sol, best_score, generate_neighbour(), evaluate_solution(), iterations, taboo_list_size, taboo_duration)")
    #Cargar heuristicas, retornar resultados de cada una.
    return resultadoTS

#Funcion para cargar componentes genericos, requiere sandboxing.
def cargarComponente(codigo: str):
    variablesLocales: Dict[str, Any] = {}
    try:
        exec(codigo, globals(), variablesLocales)
    except Exception as e:
        raise RuntimeError(f"Error al ejecutar el código (exec): {e}")
    for key, value in variablesLocales.items():
        if callable(value):
            return value
    raise ValueError(f"No se encontró una función (callable) en el código cargado con exec().")

def cronometrarFuncion(func: Callable, *args, **kwargs) -> tuple[float, any]:
    inicio = time.perf_counter()
    resultado = func(*args, **kwargs)
    fin = time.perf_counter()
    tiempo = fin - inicio
    return resultado, tiempo


def guardarResultado(problemaID,resultDB, representacion, evaluacion, vecindad, perturbacion, resultados,mejorSolucion, optimo, MH, tiempo):
    datosResultado = {
        'ID_Problema': problemaID,
        'Representacion': representacion,
        'Evaluacion': evaluacion,
        'Vecindad': vecindad,
        'Perturbacion': perturbacion,
        'Resultados': resultados,
        'Solucion': mejorSolucion,
        'Valor Optimo': optimo,
        'Metaheuristica': MH,
        'Tiempo': tiempo
    }
    dfAux = pd.DataFrame([datosResultado])
    resultDBMod = pd.concat([resultDB,dfAux], ignore_index=True)
    return resultDBMod
def guardarFeedback(problemaID,feedbackDB, representacion, componentes, version, feedback):
    datosFeedback = {
        'ID_Problema': problemaID,
        'Representacion': representacion,
        'Componentes': componentes,
        'Version': version,
        'Feedback': feedback
    }
    dfAux = pd.DataFrame([datosFeedback])
    feedbackDBMod = pd.concat([feedbackDB,dfAux], ignore_index=True)
    return feedbackDBMod
def guardarComponentes(problemaID,componenteDB, representacion, evaluacion, vecindad, perturbacion, version):
    datosComponentes = {
        'ID_Problema': problemaID,
        'Representacion': representacion,
        'Evaluacion': evaluacion,
        'Vecindad': vecindad,
        'Perturbacion': perturbacion,
        'Version': version
    }
    dfAux = pd.DataFrame([datosComponentes])
    componenteDBMod = pd.concat([componenteDB,dfAux], ignore_index=True)
    return componenteDBMod

main()