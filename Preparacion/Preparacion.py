# Modulo integrador, contiene los loops de la arquitectura general
from typing import Dict, Any
from Instancias import NLInstance,DataLoader
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

def extraerProblema(llms:generador,instancia:NLInstance):
    ## Generar prompts con PromptSamplerDM
    prompt = generateSeedPrompt(instancia.problem)
    ## Prompt con tipo de problema. Este es menos exigente en las capacidades de razocinio de la LLM. Servira para pruebas a futuro
    #prompt = generateSeedPromptWithProblemTYpe(instanciaPrueba.problem,instanciaPrueba.problemType,semilla)
    ## Preparar enjambre de LLMs (Actualmente solo con ChatGPT)
    respuesta = llms.extraccionDatos(prompt)
    return respuesta


def evaluarExtraccion(llms: generador,instancia:NLInstance, respuesta):
    prompt = generateFeedbackPrompt(instancia.problem,respuesta,instancia.parsedSolution,instancia.objectiveScore)
    feedback = llms.extraccionDatos(prompt)
    return feedback

def refinarDescripcion(llms: generador,instancia:NLInstance,feedback:str):
    prompt = updatePrompt(instancia.problem, instancia.problemType,instancia.objectiveScore,feedback)
    respuesta = llms.extraccionDatos(prompt)
    return respuesta

def extraerIndividual(dataloader:DataLoader,llms:generador,path,dataStructPath, ID:str, schema:str,dataclassProblema):
    tiempoInicio = time.perf_counter()
    instancias = dataloader.getInstancias(ID)
    latestFeedback = "NA"
    header = ['Instancia','Traje','Tipo_de_problema', 'Subtipo_de_problema', 'Iteracion', 'Respuesta', 'Feedback', 'Datos', 'Resultado_esperado', 'Valor_Objetivo', 'tiempo']
    for instancia in instancias:
        i = 1
        mejorSolucion = instancia.parsedSolution
        valorOptimo = instancia.objectiveScore
        respuesta = extraerProblema(llms,instancia)
        latestResponse = respuesta.content[0]['text']
        #Se assume que esta correcto. Si falla cargar resultados, ahi recien se itera
        try:
            extractedSchema = json.loads(respuesta.content[0]['text'])
            convertidorText = generarDataClass(llms, extractedSchema["DATA_ROLES"], extractedSchema, schema)
            _cargarDatosProblema(extractedSchema,dataclassProblema, convertidorText) #presente solo para saber si funciona, Un smoke test en essencia
        except Exception as e: #La prueba consiste en cargar los datos a memoria usando el dataclass del plugin. Si funciona sabemos que 1 se puede cargar, 2 los datos son lo suficientemente validos para generar una solucion.  
            print("Excepcion. Primer intento fallido")  
            feedback = evaluarExtraccion(llms,instancia, respuesta.content[0]['text']) # podriamos darle el contenido de le excepcion
            respuesta = refinarDescripcion(llms,instancia,feedback)
            extractedSchema = json.loads(respuesta.content[0]['text'])
            convertidorText = generarDataClass(llms, extractedSchema["DATA_ROLES"], extractedSchema, schema)
            latestFeedback = feedback.content[0]['text'] 
            latestResponse = respuesta.content[0]['text']
            i = 2
        ## con esto solo guarda lo que funciono. I lleva cuenta de la cantidad de intentos que tomo. 
        datos = [instancia.claveInstancia, instancia.problemCostume, instancia.problemType, instancia.problemSubType, i, latestResponse, latestFeedback, instancia.instanceContent, instancia.parsedSolution, instancia.objectiveScore, time.perf_counter()-tiempoInicio]   
        guardarResultados(datos, header, path)
        dataStructDatos = [instancia.claveInstancia,instancia.problemType,extractedSchema,convertidorText]
        dataStructHeader = ['Instancia','Tipo_de_problema','SchemaDataClass','FuncionDeCarga']
        guardarResultados(dataStructDatos, dataStructHeader, dataStructPath)
        
    return latestResponse, latestFeedback, convertidorText

##Estos dos estan repetidos en optimizacion. hay que moverlos a un archivo aparte, porque los usa main. Aux?. lib?

def _cargarDatosProblema(problemData, standardDataclass, convertidorText):
    convertidor = _cargarComponente(convertidorText)
    problemaActual = convertidor(problemData,standardDataclass)
    return problemaActual

def _cargarComponente(codigo: str):
    variablesLocales: Dict[str, Any] = {}
    try:
        exec(codigo, globals(), variablesLocales)
    except Exception as e:
        raise RuntimeError(f"Error al ejecutar el código (exec): {e}")
    for key, value in variablesLocales.items():
        if callable(value):
            return value
    raise ValueError(f"No se encontró una función (callable) en el código cargado con exec().")



def generarDataClass(llms: generador,extractedData,muestra, schema):
    prompt = generateConverterPrompt(extractedData, muestra, schema)
    respuesta = llms.extraccionDatos(prompt)
    return respuesta.content[0]['text']

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