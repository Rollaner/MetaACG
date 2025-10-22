# Modulo integrador, contiene los loops de la arquitectura general
from .InstanciasProblemas import Instancia,DataLoader
from Generador import generador
from .PromptSamplerDM import *
import time
import csv

#Recalibración de las LLMS
def reiniciarLLMS():
    llms = generador()
    llms.cargarLLMs()
    return llms

def definirBatch(instancias:DataLoader,llms:generador, path, tipo:str):
    tiempoInicio = time.perf_counter()
    header = ['Instancia','Traje','Tipo de problema', 'Subtipo de problema', 'Iteracion', 'Respuesta', 'Feedback', 'Resultado esperado', 'tokens', 'tiempo']
    csvDefinicion = path
    i = 0
    try:
        with open(csvDefinicion, 'w', newline='', encoding='utf-8') as csvfile:
                escritor = csv.writer(csvfile)
                escritor.writerow(header)
        with open(csvDefinicion, 'a', newline='', encoding='utf-8') as csvfile:
            escritor = csv.writer(csvfile)
            for instancia in instancias.getAllInstancias():
                if instancia.problemType != tipo:
                    continue
                respuesta = definirProblema(llms,instancia)
                datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, i, respuesta.content[0]['text'], '', '', time.perf_counter()-tiempoInicio]
                escritor.writerow(datos)
                print(instancia.claveInstancia, respuesta.content[0]['text'])
                for j in range(2):
                    feedback, resultados = evaluarDefinicion(llms,instancia, respuesta.content[0]['text'])
                    datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, j, respuesta.content[0]['text'], feedback, resultados, time.perf_counter()-tiempoInicio]         
                    respuesta = refinarDefinicion(llms,instancia,feedback, resultados)
                    escritor.writerow(datos)
                    print(instancia.claveInstancia, respuesta.content[0]['text'])
                datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, i, respuesta.content[0]['text'], '', '', time.perf_counter()-tiempoInicio]
                escritor.writerow(datos)
                print(instancia.claveInstancia, respuesta.content[0]['text'])
                reiniciarLLMS()
                i = i + 1
    except Exception as e:
                print(f"Error de escritura durante el proceso de definicion, no se han guardado los resultados {e}")
    print("Fin proceso de definicion matematica")

def definirProblema(llms,instancia:Instancia):
    ## Generar prompts con PromptSamplerDM
    prompt = generateSeedPrompt(instancia.problem)
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
    prompt = generateFeedbackPromptNR(instancia.problem,respuestaDef,respuestaObj,respuestaEval,instancia.solutionValue)
    feedback = llms.generarFeedback(prompt)
    return feedback, instancia.solutionValue

def refinarDefinicion(llms,instancia:Instancia,feedback:str, resultados):
    prompt = updatePrompt(instancia.problem, instancia.problemType,resultados,instancia.solutionValue,feedback)
    respuesta = llms.generarDefinicion(prompt)
    return respuesta

def definirIndividual(instancias:DataLoader,llms:generador,path, ID:str):
    tiempoInicio = time.perf_counter()
    instancia:Instancia = instancias.getDatosInstancia(ID)
    respuesta = definirProblema(llms,instancia)
    csvDefinicion = path
    with open(csvDefinicion, 'a', newline='', encoding='utf-8') as csvfile:
            escritor = csv.writer(csvfile)
            datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, "i", respuesta.content[0]['text'], '', '', time.perf_counter()-tiempoInicio]
            escritor.writerow(datos)
            for j in range(2):
                feedback, resultados = evaluarDefinicion(llms,instancia, respuesta.content[0]['text'])
                datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, "i"+j, respuesta.content[0]['text'], feedback, resultados, time.perf_counter()-tiempoInicio]         
                respuesta = refinarDefinicion(llms,instancia,feedback, resultados)
                escritor.writerow(datos)
            datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, "i", respuesta.content[0]['text'], '', '', time.perf_counter()-tiempoInicio]
            escritor.writerow(datos)
    return respuesta, feedback, resultados