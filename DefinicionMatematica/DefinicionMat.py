# Modulo integrador, contiene los loops de la arquitectura general
from Instancias import Instancia,DataLoader
from Generador import generador
from .PromptSamplerDM import *
import time
import sys
import json

#Recalibración de las LLMS
def reiniciarLLMSDef():
    llms = generador()
    llms.cargarLLMs()
    return llms

def definirBatch(instancias:DataLoader,llms:generador, path, tipo:str):
    tiempoInicio = time.perf_counter()
    header = ['Instancia','Traje','Tipo de problema', 'Subtipo de problema', 'Iteracion', 'Respuesta', 'Feedback', 'Resultado esperado','Valor Objetivo', 'tiempo']
    for instancia in instancias.getAllInstancias():
        respuesta = definirProblema(llms,instancia)
        datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, 0, respuesta.content[0]['text'], 'None',instancia.parsedSolution, instancia.objectiveScore, time.perf_counter()-tiempoInicio]
        guardarResultados(datos, header, path)
        print(instancia.claveInstancia, respuesta.content[0]['text'])
        for i in range(2):
            feedback = evaluarDefinicion(llms,instancia, respuesta.content[0]['text'])
            datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, i+1, respuesta.content[0]['text'], feedback.content[0]['text'],instancia.parsedSolution, instancia.objectiveScore, time.perf_counter()-tiempoInicio]         
            respuesta = refinarDefinicion(llms,instancia,feedback,instancia.objectiveScore)
            guardarResultados(datos, header, path)
            print(instancia.claveInstancia, respuesta.content[0]['text'])
        reiniciarLLMSDef()
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
    prompt = generateFeedbackPromptNR(instancia.problem,respuestaDef,respuestaObj,respuestaEval,instancia.parsedSolution,instancia.objectiveScore)
    feedback = llms.generarFeedback(prompt)
    return feedback

def refinarDefinicion(llms,instancia:Instancia,feedback:str, resultados):
    prompt = updatePrompt(instancia.problem, instancia.problemType,resultados,instancia.objectiveScore,feedback)
    respuesta = llms.generarDefinicion(prompt)
    return respuesta

def definirIndividual(instancias:DataLoader,llms:generador,path, ID:str):
    tiempoInicio = time.perf_counter()
    instancia:Instancia = instancias.getDatosInstancia(ID)
    header = ['Instancia','Traje','Tipo de problema', 'Subtipo de problema', 'Iteracion', 'Respuesta', 'Feedback', 'Resultado esperado', 'Valor Objetivo', 'tiempo']
    respuesta = definirProblema(llms,instancia)
    datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, "i", respuesta.content[0]['text'], 'None', 'None', time.perf_counter()-tiempoInicio]
    guardarResultados(datos, header, path)
    for j in range(2):
        feedback, resultados = evaluarDefinicion(llms,instancia, respuesta.content[0]['text'])
        datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, "i"+j, respuesta.content[0]['text'], feedback, resultados, time.perf_counter()-tiempoInicio]         
        respuesta = refinarDefinicion(llms,instancia,feedback, resultados)
        guardarResultados(datos, header, path)
    datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, "i", respuesta.content[0]['text'], 'None', 'None', time.perf_counter()-tiempoInicio]
    guardarResultados(datos, header, path)
    return respuesta, feedback, resultados

def guardarResultados(datos, header, path):
    num_encabezados = len(header)
    num_datos = len(datos)
    if num_datos != num_encabezados:
        print(
            f"Error: La cantidad de datos ({num_datos}) no encajan con el numero de encabezados ({num_encabezados})."
            f"El registro NO fue guardado. Use 'None' explícito para campos faltantes."
            ,file=sys.stderr)
        return
    record_dict = dict(zip(header, datos))
    json_line = json.dumps(record_dict, ensure_ascii=False)

    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json_line + '\n')
    except Exception as e:
        print(f"Ocurrio un error al momento de escribir en el archivo: {e}", file=sys.stderr)