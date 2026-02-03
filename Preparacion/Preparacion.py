# Modulo integrador, contiene los loops de la arquitectura general

from Instancias import InstanciaEHOP,DataLoader
from Generador import generador
from .PromptSamplerP import *
import time
import sys
import json

#Recalibración de las LLMS
def reiniciarLLMSDef():
    llms = generador()
    llms.cargarLLMs()
    return llms

def prepararBatch(instancias:DataLoader,llms:generador, path, tipo:str):
    tiempoInicio = time.perf_counter()
    header = ['Instancia','Traje','Tipo de problema', 'Subtipo de problema', 'Iteracion', 'Respuesta', 'feedback','Datos', 'Resultado esperado','Valor Objetivo', 'tiempo']
    for instancia in instancias.getAllInstancias():
        respuesta = extraerProblema(llms,instancia)
        if instancia.problemSubType == "Inverted":
            continue
        else:
            mejorSolucion = instancia.parsedSolution
            valorOptimo = instancia.objectiveScore
        datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, 0, respuesta.content[0]['text'],"NA", instancia.instanceContent,mejorSolucion, valorOptimo, time.perf_counter()-tiempoInicio]
        guardarResultados(datos, header, path)
        print(instancia.claveInstancia, respuesta.content[0]['text'])
        for i in range(2):
            feedback = evaluarExtraccion(llms,instancia, respuesta.content[0]['text'])
            datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, i+1, respuesta.content[0]['text'], feedback.content[0]['text'], instancia.instanceContent,instancia.parsedSolution, instancia.objectiveScore, time.perf_counter()-tiempoInicio]         
            respuesta = refinarDescripcion(llms,instancia,feedback)
            guardarResultados(datos, header, path)
            print(instancia.claveInstancia, respuesta.content[0]['text'])
        reiniciarLLMSDef()
    print("Fin proceso de definicion matematica")

def prepararSinDefinir(instancias:DataLoader,llms:generador, path, tipo:str):
    tiempoInicio = time.perf_counter()
    header = ['Instancia','Traje','Tipo de problema', 'Subtipo de problema', 'Iteracion', 'Respuesta', 'Datos', 'Resultado esperado','Valor Objetivo', 'tiempo']
    for instancia in instancias.getAllInstancias():
        datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, 0, instancia.problem, instancia.instanceContent,instancia.parsedSolution, instancia.objectiveScore, time.perf_counter()-tiempoInicio]
        guardarResultados(datos, header, path)
    print("Fin proceso de definicion matematica")

def extraerProblema(llms,instancia:InstanciaEHOP):
    ## Generar prompts con PromptSamplerDM
    prompt = generateSeedPrompt(instancia.problem)
    ## Prompt con tipo de problema. Este es menos exigente en las capacidades de razocinio de la LLM. Servira para pruebas a futuro
    #prompt = generateSeedPromptWithProblemTYpe(instanciaPrueba.problem,instanciaPrueba.problemType,semilla)
    ## Preparar enjambre de LLMs (Actualmente solo con ChatGPT)
    respuesta = llms.generarDefinicion(prompt)
    return respuesta


def evaluarExtraccion(llms,instancia:InstanciaEHOP, respuesta):
    valores = []
    for valor in respuesta.split(','):
        valores.append(valor.strip('"'))
    extractedData = valores [1]
    respuestaObj = valores[2]
    #respuestaEval = valores[3]
    prompt = generateFeedbackPrompt(instancia.problem,extractedData,instancia.parsedSolution,instancia.objectiveScore)
    feedback = llms.generarFeedback(prompt)
    return feedback

def refinarDescripcion(llms,instancia:InstanciaEHOP,feedback:str):
    prompt = updatePrompt(instancia.problem, instancia.problemType,instancia.objectiveScore,feedback)
    respuesta = llms.generarDefinicion(prompt)
    return respuesta

def extraerIndividual(dataloader:DataLoader,llms:generador,path,dataStructPath, ID:str):
    tiempoInicio = time.perf_counter()
    instancias = dataloader.getInstancias(ID)
    header = ['Instancia','Traje','Tipo_de_problema', 'Subtipo_de_problema', 'Iteracion', 'Respuesta', 'Feedback', 'Datos', 'Resultado_esperado', 'Valor_Objetivo', 'tiempo']
    for instancia in instancias:
        if instancia.problemSubType == "Inverted":
        ## hay que implementar logica para problemas invertidos!
            return
        else:
            mejorSolucion = instancia.parsedSolution
            valorOptimo = instancia.objectiveScore
        respuesta = extraerProblema(llms,instancia)
        datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, 0, respuesta.content[0]['text'],"NA", instancia.instanceContent,mejorSolucion, valorOptimo, time.perf_counter()-tiempoInicio]
        guardarResultados(datos, header, path)
        print(instancia.claveInstancia, respuesta.content[0]['text'])
        for i in range(2):
            feedback = evaluarExtraccion(llms,instancia, respuesta.content[0]['text'])
            datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, i+1, respuesta.content[0]['text'], feedback.content[0]['text'], instancia.instanceContent, instancia.parsedSolution, instancia.objectiveScore, time.perf_counter()-tiempoInicio]         
            respuesta = refinarDescripcion(llms,instancia,feedback)
            guardarResultados(datos, header, path)
            print(instancia.claveInstancia, respuesta.content[0]['text'])
        #Asi solo guarda respuestas finales
        extractedSchema = json.loads(respuesta.content[0]['text'])
        convertidor = generarDataClass(llms, extractedSchema["DATA_ROLES"],dataloader.getTestData()[0]).content[0]['text']
        datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, i+1, respuesta.content[0]['text'], feedback.content[0]['text'], instancia.instanceContent, instancia.parsedSolution, instancia.objectiveScore, time.perf_counter()-tiempoInicio]
        guardarResultados(datos, header, path)
        dataStructDatos = [instancia.claveInstancia,instancia.problemType,extractedSchema,convertidor]
        dataStructHeader = ['Instancia','Tipo_de_problema','SchemaDataClass','FuncionDeCarga']
        guardarResultados(dataStructDatos, dataStructHeader, dataStructPath)
    return  datos[5], datos[6], convertidor


def generarDataClass(llms,schema,muestra):
    prompt = generateConverterPrompt(schema, muestra)
    respuesta = llms.generarDefinicion(prompt)
    return respuesta

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