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
from typing import List, Union
from DefinicionMatematica import DefinicionMat
from Instancias import DataLoader
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
        if os.path.exists(componentesPath):
            componenteDB = pd.read_json(componentesPath,lines=True)
        else:
            componenteDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Evaluacion', 'Vecindad', 'Perturbacion','Version'])
        problema = PromptSamplerOP.sampleProblemaDB(problemaDB,semilla)
        ProblemaID, seedPrompt = PromptSamplerOP.generateSeedPrompt(problema, semilla)
        llms = generador()
        llms.cargarLLMs()
        respuestaInicial = llms.generarComponentes(seedPrompt)
        textoBruto = respuestaInicial.content[0]['text']
        componentes = json.loads(textoBruto)
        print(componentes)
        #feedbackDB = pd.read_json(feedbackPath,lines=True)
        #resultDB = pd.read_json(resultPath,lines=True)
        #optimizarProblemaAleatorio(problemaDB,componenteDB,resultDB,feedbackDB,semilla)
        #componenteDB.to_json(componentesPath,lines=True)
        #feedbackDB.to_json(feedbackPath,lines=True)
        #resultDB.to_json(resultPath,lines=True)
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
def optimizarProblemaAleatorio(problemaDB:pd.DataFrame,componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame,seed):
    problema = PromptSamplerOP.sampleProblemaDB(problemaDB,seed)
    seedPrompt = PromptSamplerOP.generateSeedPrompt(problema, seed)
    respuestas = []
    while iterations > 0:
        resultado = evaluarComponentes(respuesta)
        resultDB.add(resultado)
        feedbackPrompt = PromptSamplerOP.generateFeedbackPrompt(problema,respuesta,resultado)
        feedback = llms.generarFeedback(feedbackPrompt) 
        feedbackDB.add(feedback)
        newPrompt = PromptSamplerOP.updatePrompt(problema,componenteDB,resultDB,feedback,seed)
        respuesta = llms.generarComponentes(newPrompt)
        respuestas.append(respuesta)
        componenteDB.add(respuesta)
        iterations = iterations - 1
        if iterations % 3 == 0:
            llms = DefinicionMat.reiniciarLLMS()

def optimizarProblemaPredefinido(problema,componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame,seed):
    seedPrompt = PromptSamplerOP.generateSeedPrompt(problema,componenteDB, seed)
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
        feedbackPrompt = PromptSamplerOP.generateFeedbackPrompt(problema,respuesta.content[0]['text'],resultado)
        feedback = llms.generarFeedback(feedbackPrompt) 
        feedbackDB.add(feedback)
        newPrompt = PromptSamplerOP.updatePrompt(problema,componenteDB,resultDB,feedback,seed)
        respuesta = llms.generarComponentes(newPrompt)
        respuestas.append(respuesta.content[0]['text'])
        componenteDB.add(respuesta.content[0]['text'])
        iterations = iterations - 1
        if iterations % 3 == 0:
            llms = DefinicionMat.reiniciarLLMS()
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