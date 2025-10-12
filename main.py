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
    #Datos inicializados
    respuesta, instancia = definirProblema(instancias)
    print(respuesta)
    feedback = evaluarDefinicion(instancia, respuesta)
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
    return feedback

def probarDefinicion(objective:str,eval:str,parsedSolution:list):
    ## viendo como reducir el riesgo de las alucinaciones. Son poco probables, pero igual por si acaso
    ## restrictedPyhton es demasiado viego. ast_eval() es probablemente la mejor solucion. We have to provide a dict for the parsed solution, and an empty dict of global variables
    ## Then we remove the builtins. It's not bulletproof but should grab MOST of the jailbreak attempts. No open, no write, etc. {"__builtins__":None},safe_dict.
    ## as the code should only need the solution to evaluate (and return a mere number) it should work fine.
    return 0

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