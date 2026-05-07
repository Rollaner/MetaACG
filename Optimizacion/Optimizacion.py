import ast
import inspect
import linecache
import time
import numbers
from typing import Any, Callable, Dict, Tuple,List, Union
from dataclasses import make_dataclass
from Instancias import NLInstance,DataLoader
from Generador import generador
from .PromptSamplerOP import *
import Heuristicas.IteratedLocalSearch
import Heuristicas.SimulatedAnnealing
import Heuristicas.TabooSearch
import Heuristicas.HillClimbing
import traceback
import sys
import json
import numpy as np
import re
import math
import random
from functools import partial

#Recalibración de las LLMS
def reiniciarLLMS():
    llms = generador()
    llms.cargarLLMs()
    return llms

def cronometrarFuncion(func: Callable, *args, **kwargs) -> tuple[float, any]:
    inicio = time.perf_counter()
    resultado = func(*args, **kwargs)
    fin = time.perf_counter()
    tiempo = fin - inicio
    return resultado, tiempo

heuristicas = [
    (Heuristicas.SimulatedAnnealing.SA,[1000,10,0.9772],"SA","Simulated_Annealing"),
    (Heuristicas.IteratedLocalSearch.ILS,[200,0.1],"ILS","Iterated_Local_Search"),
    (Heuristicas.TabooSearch.TS, [200,10,7],"TS","Taboo_Search"),
    (Heuristicas.HillClimbing.HC, [200],"HC","Hill_Climbing")
]


def extraerDatosProblema(problemaSample:pd.DataFrame):
    problemaID = problemaSample.Instancia
    inspiraciones=json.loads(problemaSample.Respuesta)
    knownSol = problemaSample.Resultado_esperado
    knownObj = problemaSample.Valor_Objetivo
    return problemaID, inspiraciones, knownSol, knownObj

def generacionComponentesInicial(schema,objetivo, restricciones, llms, sampleSol,inspiracionEval,inspiracionPerturb,inspiracionNB):
    promptEval = generateEvalPrompt(schema,objetivo, restricciones, sampleSol,inspiracionEval)
    promptNb = generateNBPrompt(schema,objetivo, restricciones, sampleSol,inspiracionNB)
    promptPerturb = generatePerturbPrompt(schema,objetivo, restricciones, sampleSol,inspiracionPerturb)
    RawEval = llms.generarComponente(promptEval)
    RawNb = llms.generarComponente(promptNb)
    RawPerturb = llms.generarComponente(promptPerturb)
    textoBruto = RawEval.content[0]['text']
    Eval = json.loads(textoBruto)
    textoBruto = RawNb.content[0]['text']
    Nb = json.loads(textoBruto)
    textoBruto = RawPerturb.content[0]['text']
    Perturb = json.loads(textoBruto)

    EvalTokenInput = RawEval.usage_metadata.get("input_tokens")
    EvalTokenOutput = RawEval.usage_metadata.get("output_tokens")
    EvalTokenTotal = RawEval.usage_metadata.get("total_tokens")

    PerturbTokenInput = RawPerturb.usage_metadata.get("input_tokens")
    PerturbTokenOutput = RawPerturb.usage_metadata.get("output_tokens")
    PerturbTokenTotal = RawPerturb.usage_metadata.get("total_tokens")

    NbTokenInput = RawNb.usage_metadata.get("input_tokens")
    NbTokenOutput = RawNb.usage_metadata.get("output_tokens")
    NbTokenTotal = RawNb.usage_metadata.get("total_tokens")
 
    return Eval, Nb, Perturb, Eval['REPRESENTATION'], EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal

def generacionNuevosComponentes(schema,objetivo, restricciones, llms, sampleSol,feedbackEval,feedbackNb,feedbackPerturb,inspiracionEval,inspiracionPerturb,inspiracionNB) -> Tuple[Any,Any,Any]: ## Requiere TLC para que considere todos los resultados. Mezclemolos en uno. 
    #Hay que hacer que revisa con cada tipo de resultado. No solo Simmulated Annealing
    promptEval = updateEvalPrompt(schema,objetivo, restricciones, feedbackEval, sampleSol,inspiracionEval)
    promptNb = updateNBPrompt(schema,objetivo, restricciones, feedbackNb, sampleSol,inspiracionNB)
    promptPerturb = updatePerturbPrompt(schema,objetivo, restricciones, feedbackPerturb, sampleSol,inspiracionPerturb)
    RawEval = llms.generarComponente(promptEval)
    RawNb = llms.generarComponente(promptNb)
    RawPerturb = llms.generarComponente(promptPerturb)
    textoBruto = RawEval.content[0]['text']
    Eval = json.loads(textoBruto)
    textoBruto = RawNb.content[0]['text']
    Nb = json.loads(textoBruto)
    textoBruto = RawPerturb.content[0]['text']
    Perturb = json.loads(textoBruto)
    print(Perturb['PERTURB_CODE'])
    
    EvalTokenInput = RawEval.usage_metadata.get("input_tokens")
    EvalTokenOutput = RawEval.usage_metadata.get("output_tokens")
    EvalTokenTotal = RawEval.usage_metadata.get("total_tokens")

    PerturbTokenInput = RawPerturb.usage_metadata.get("input_tokens")
    PerturbTokenOutput = RawPerturb.usage_metadata.get("output_tokens")
    PerturbTokenTotal = RawPerturb.usage_metadata.get("total_tokens")

    NbTokenInput = RawNb.usage_metadata.get("input_tokens")
    NbTokenOutput = RawNb.usage_metadata.get("output_tokens")
    NbTokenTotal = RawNb.usage_metadata.get("total_tokens")

    return Eval, Nb, Perturb, Eval['REPRESENTATION'], EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal

def generacionComponentesInicialSP(problema, llms, sampleSol,inspiracionEval,inspiracionPerturb,inspiracionNB):
    promptEval = generateEvalPromptSP(problema, sampleSol,inspiracionEval)
    promptNb = generateNBPromptSP(problema, sampleSol,inspiracionNB)
    promptPerturb = generatePerturbPromptSP(problema, sampleSol,inspiracionPerturb)
    RawEval = llms.generarComponente(promptEval)
    RawNb = llms.generarComponente(promptNb)
    RawPerturb = llms.generarComponente(promptPerturb)
    textoBruto = RawEval.content[0]['text']
    Eval = json.loads(textoBruto)
    textoBruto = RawNb.content[0]['text']
    Nb = json.loads(textoBruto)
    textoBruto = RawPerturb.content[0]['text']
    Perturb = json.loads(textoBruto)
    EvalTokenInput = RawEval.usage_metadata.get("input_tokens")
    EvalTokenOutput = RawEval.usage_metadata.get("output_tokens")
    EvalTokenTotal = RawEval.usage_metadata.get("total_tokens")

    PerturbTokenInput = RawPerturb.usage_metadata.get("input_tokens")
    PerturbTokenOutput = RawPerturb.usage_metadata.get("output_tokens")
    PerturbTokenTotal = RawPerturb.usage_metadata.get("total_tokens")

    NbTokenInput = RawNb.usage_metadata.get("input_tokens")
    NbTokenOutput = RawNb.usage_metadata.get("output_tokens")
    NbTokenTotal = RawNb.usage_metadata.get("total_tokens")

    return Eval, Nb, Perturb, Eval['REPRESENTATION'], EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal

def generacionNuevosComponentesSP(problema, llms, sampleSol,feedbackEval,feedbackNb,feedbackPerturb,inspiracionEval,inspiracionPerturb,inspiracionNB) -> Tuple[Any,Any,Any]: ## Requiere TLC para que considere todos los resultados. Mezclemolos en uno. 
    #Hay que hacer que revisa con cada tipo de resultado. No solo Simmulated Annealing
    promptEval = updateEvalPromptSP(problema,  feedbackEval, sampleSol,inspiracionEval)
    promptNb = updateNBPromptSP(problema, feedbackNb, sampleSol,inspiracionNB)
    promptPerturb = updatePerturbPromptSP(problema, feedbackPerturb, sampleSol,inspiracionPerturb)
    RawEval = llms.generarComponente(promptEval)
    RawNb = llms.generarComponente(promptNb)
    RawPerturb = llms.generarComponente(promptPerturb)
    textoBruto = RawEval.content[0]['text']
    Eval = json.loads(textoBruto)
    textoBruto = RawNb.content[0]['text']
    Nb = json.loads(textoBruto)
    textoBruto = RawPerturb.content[0]['text']
    Perturb = json.loads(textoBruto)
    print(Perturb['PERTURB_CODE'])
    EvalTokenInput = RawEval.usage_metadata.get("input_tokens")
    EvalTokenOutput = RawEval.usage_metadata.get("output_tokens")
    EvalTokenTotal = RawEval.usage_metadata.get("total_tokens")

    PerturbTokenInput = RawPerturb.usage_metadata.get("input_tokens")
    PerturbTokenOutput = RawPerturb.usage_metadata.get("output_tokens")
    PerturbTokenTotal = RawPerturb.usage_metadata.get("total_tokens")

    NbTokenInput = RawNb.usage_metadata.get("input_tokens")
    NbTokenOutput = RawNb.usage_metadata.get("output_tokens")
    NbTokenTotal = RawNb.usage_metadata.get("total_tokens")

    return Eval, Nb, Perturb, Eval['REPRESENTATION'], EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal

def generarFeedback(llms, prompt):
    rawFeedback = llms.generarFeedback(prompt)
    feedbackText = rawFeedback.content[0]['text'] 
    feedbackTokenInput = rawFeedback.usage_metadata.get("input_tokens")
    feedbackTokenOutput = rawFeedback.usage_metadata.get("output_tokens")
    feedbackTokenTotal = rawFeedback.usage_metadata.get("total_tokens")
    return feedbackText,feedbackTokenInput,feedbackTokenOutput,feedbackTokenTotal



def guardarResultados(problemaID, i, representacion, Eval, Nb, Perturb, knownSol, status,EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal, *args:list):
    
    for arg in args:
        resultado,tiempo,heuristica,_ = arg
        resultDB = guardarResultado(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,resultado,knownSol, "NA", heuristica,status, tiempo,EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal)
    return resultDB

## Detecta si todo obtuvo resultados validos, y si alguna de las metaheuristicas llego al optimo. Si una lo hizo, sabemos que se puede, y que los componententes estan correctos.
def componentesCorrectos(Eval, problemData, knownSol, knownObj, *args):
    optimoPresente:bool = False
    errorNoDetectado:bool = False
    try:
        eval = cargarComponente(Eval['EVAL_CODE'], "<evaluate_solution>")
    except Exception as e:
        return generarDiagnostico(f"Failed to load EVAL_CODE: {e}")        
    try:
        puntajeCalculado = Eval(knownSol,problemData)
        #print(f"Loaded 'EVAL_CODE' into variable 'evaluate_solution'. Name: {eval.__name__}")
    except Exception as e:
        return generarDiagnostico(f"Failed to load TEST_SOL: {e}. TEST_SOL: {knownSol}"), optimoPresente, errorNoDetectado
    
    
    if puntajeCalculado != knownObj:
        return f"EVAL_CODE(TEST_SOL) desn't match expected TEST_SOL Score.", optimoPresente, errorNoDetectado
    for arg in args:
        resultados,_,_,_ = arg
        try: 
            currentSolucion,currentScore, mejorSolucion, mejorScore = resultados
        except Exception as e:
            return f"Output results not in correct format. got: {resultados}. Expected [currentSolution,currentScore, bestSolution, bestScore]", optimoPresente, errorNoDetectado
        if (not isinstance(currentSolucion, (list, tuple)) or not isinstance(currentScore, numbers.Real) or not isinstance(mejorSolucion, (list, tuple))or not isinstance(mejorScore, numbers.Real)):      
            return f"Provided data fails sanity check. {[currentSolucion, currentScore, mejorSolucion, mejorScore]}", optimoPresente, errorNoDetectado
        else: errorNoDetectado = True
            
        if (mejorScore >= knownObj):
            optimoPresente = True
    return "EVAL_CODE_PASSED_ALL_TESTS", optimoPresente, errorNoDetectado 

def componentesCorrectosSP(Eval, knownSol, knownObj, *args):
    optimoPresente:bool = False
    errorNoDetectado:bool = False
    try:
        eval = cargarComponente(Eval['EVAL_CODE'], "<evaluate_solution>")
    except Exception as e:
        return generarDiagnostico(f"Failed to load EVAL_CODE: {e}")        
    try:
        puntajeCalculado = Eval(knownSol)
        #print(f"Loaded 'EVAL_CODE' into variable 'evaluate_solution'. Name: {eval.__name__}")
    except Exception as e:
        return generarDiagnostico(f"Failed to load TEST_SOL: {e}. TEST_SOL: {knownSol}"), optimoPresente, errorNoDetectado
    
    
    if puntajeCalculado != knownObj:
        return f"EVAL_CODE(TEST_SOL) desn't match expected TEST_SOL Score.", optimoPresente, errorNoDetectado
    for arg in args:
        resultados,_,_,_ = arg
        try: 
            currentSolucion,currentScore, mejorSolucion, mejorScore = resultados
        except Exception as e:
            return f"Output results not in correct format. got: {resultados}. Expected [currentSolution,currentScore, bestSolution, bestScore]", optimoPresente, errorNoDetectado
        if (not isinstance(currentSolucion, (list, tuple)) or not isinstance(currentScore, numbers.Real) or not isinstance(mejorSolucion, (list, tuple))or not isinstance(mejorScore, numbers.Real)):      
            return f"Provided data fails sanity check. {[currentSolucion, currentScore, mejorSolucion, mejorScore]}", optimoPresente, errorNoDetectado
        else: errorNoDetectado = True
            
        if (mejorScore >= knownObj):
            optimoPresente = True
    return "EVAL_CODE_PASSED_ALL_TESTS", optimoPresente, errorNoDetectado 


def cargarDatosProblema(problemData, standardDataclass, convertidorText):
    convertidor = cargarComponente(convertidorText)
    problemaActual = convertidor(problemData,standardDataclass)
    return problemaActual

def evaluarComponentes(Eval, Nb, Perturb,problemData, SampleSol,Heuristica, Params):
    #Guardar excepciones como string, para retornarlas como debug
    if isinstance(SampleSol,str):
        try:
            solucionPrueba = ast.literal_eval(SampleSol.strip().replace(' ', ''))
        except Exception as e:
            return(generarDiagnostico(f"Failed to load SAMPLE_SOL: {e}"))
    else: solucionPrueba = SampleSol
    try:
        evaluacion = cargarComponente(Eval['EVAL_CODE'], "<evaluate_solution>")
        print(f"Loaded 'EVAL_CODE' into variable 'evaluate_solution'. Name: {eval.__name__}")
    except Exception as e:
        return(generarDiagnostico(f"Failed to load EVAL_CODE: {e}"))
    try:
        vecindad = cargarComponente(Nb['NB_CODE'],"<generate_neighbour>")
        print(f"Loaded 'NB_CODE' into variable 'generate_neighbour'. Name: {vecindad.__name__}")
    except Exception as e:
        return(generarDiagnostico(f"Failed to load NB_CODE: {e}"))
    try:
        perturb = cargarComponente(Perturb['PERTURB_CODE'], "<perturb_solution>")
        print(f"Loaded 'PERTURB_CODE' into variable 'perturb_solution'. Name: {perturb.__name__}")
    except Exception as e:
        return(generarDiagnostico(f"Failed to load PERTURB_CODE: {e}"))
    try:
        valor = evaluacion(solucionPrueba, problemData)
    except Exception as e:
        return(generarDiagnostico(f"Failed to evaluate SAMPLE_SOL with EVAL_CODE: {e}"))
    try:
        resultado = Heuristica(solucionPrueba,solucionPrueba,valor,problemData,vecindad,perturb,evaluacion,*Params)
    except Exception as e:
        return(generarDiagnostico(f"Failed to run target heuristic: {e}.  Signature def {Heuristica.__name__}(solution,best_sol, best_score, problemData, generate_neighbour(), evaluate_solution(), *Params"))
    #Cargar heuristicas, retornar resultados de cada una.
    return resultado

def evaluarComponentesSP(Heuristica,Eval, Nb, Perturb, SampleSol, Params):
    #Guardar excepciones como string, para retornarlas como debug
    if isinstance(SampleSol,str):
        try:
            solucionPrueba = ast.literal_eval(SampleSol.strip().replace(' ', ''))
        except Exception as e:
            return(generarDiagnostico(f"Failed to load SAMPLE_SOL: {e}"))
    else: solucionPrueba = SampleSol
    try:
        evaluacion = cargarComponente(Eval['EVAL_CODE'], "<evaluate_solution>")
        print(f"Loaded 'EVAL_CODE' into variable 'evaluate_solution'. Name: {eval.__name__}")
    except Exception as e:
        return(generarDiagnostico(f"Failed to load EVAL_CODE: {e}"))
    try:
        vecindad = cargarComponente(Nb['NB_CODE'],"<generate_neighbour>")
        print(f"Loaded 'NB_CODE' into variable 'generate_neighbour'. Name: {vecindad.__name__}")
    except Exception as e:
        return(generarDiagnostico(f"Failed to load NB_CODE: {e}"))
    try:
        perturb = cargarComponente(Perturb['PERTURB_CODE'], "<perturb_solution>")
        print(f"Loaded 'PERTURB_CODE' into variable 'perturb_solution'. Name: {perturb.__name__}")
    except Exception as e:
        return(generarDiagnostico(f"Failed to load PERTURB_CODE: {e}"))
    try:
        valor = evaluacion(solucionPrueba)
    except Exception as e:
        return(generarDiagnostico(f"Failed to evaluate SAMPLE_SOL with EVAL_CODE: {e}"))
    try:
        resultado = Heuristica(solucionPrueba,solucionPrueba,valor,vecindad,perturb,evaluacion,*Params)
    except Exception as e:
        return(generarDiagnostico(f"Failed to run target heuristic: {e}.  Signature def {Heuristica.__name__}(solution,best_sol, best_score, problemData, generate_neighbour(), evaluate_solution(), *Params"))
    #Cargar heuristicas, retornar resultados de cada una.
    return resultado


#Funcion para cargar componentes genericos, requiere sandboxing.
def cargarComponente(codigo: str, nombre: str = "<componente>"):
    variablesLocales: Dict[str, Any] = {}
    filename = f"<{nombre}>"
    lines = codigo.splitlines(keepends=True)
    linecache.cache[filename] = (
        len(codigo),
        None,
        lines,
        filename
    )
    
    try:
        exec(compile(codigo, filename, "exec"), globals(), variablesLocales)
    except Exception as e:
        raise RuntimeError(f"Error al ejecutar el código (exec): {e}")
    for key, value in variablesLocales.items():
        if callable(value):
            return value
    raise ValueError(f"No se encontró una función (callable) en el código cargado con exec().")

def guardarResultado(problemaID, iteracion,resultDB, representacion, evaluacion, vecindad, perturbacion, resultados,mejorSolucion, optimo, MH,status, tiempo, EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal):
    datosResultado = {
        'ID_Problema': problemaID,
        'Iteracion': iteracion,
        'Representacion': representacion,
        'Evaluacion': evaluacion,
        'Vecindad': vecindad,
        'Perturbacion': perturbacion,
        'Resultados': resultados,
        'Solucion': mejorSolucion,
        'Valor Optimo': optimo,
        'Metaheuristica': MH,
        'Status': status,
        'Tiempo': tiempo,
        'TokensEvalIn' : EvalTokenInput,
        'TokensEvalOut' : EvalTokenOutput,
        'TokensEvalTot' : EvalTokenTotal,
        'TokensNbIn' : NbTokenInput,
        'TokensNbOut' : NbTokenOutput,
        'TokensNbTot' : NbTokenTotal,
        'TokensPerturbIn' : PerturbTokenInput,
        'TokensPertubOut' : PerturbTokenOutput,
        'TokensPerturbTot' : PerturbTokenTotal
    }
    dfAux = pd.DataFrame([datosResultado])
    dfAux = dfAux.dropna(axis=1, how='all')
    resultDBMod = pd.concat([resultDB,dfAux], ignore_index=True) 
    return resultDBMod

def guardarFeedback(problemaID,feedbackDB, representacion, Eval, Nb, Perturb, version, feedbackEval, feedbackNb, feedbackPerturb,EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal):
    datosFeedback = {
        'ID_Problema': problemaID,
        'Representacion': representacion,
        'Evaluacion': Eval,
        'Vecindad': Nb,
        'Perturbacion': Perturb,
        'Version': version,
        'FeedbackEval': feedbackEval,
        'FeedbackNb': feedbackNb,
        'FeedbackPerturbn': feedbackPerturb,
        'TokensEvalIn' : EvalTokenInput,
        'TokensEvalOut' : EvalTokenOutput,
        'TokensEvalTot' : EvalTokenTotal,
        'TokensNbIn' : NbTokenInput,
        'TokensNbOut' : NbTokenOutput,
        'TokensNbTot' : NbTokenTotal,
        'TokensPerturbIn' : PerturbTokenInput,
        'TokensPertubOut' : PerturbTokenOutput,
        'TokensPerturbTot' : PerturbTokenTotal
    }
    dfAux = pd.DataFrame([datosFeedback])
    feedbackDBMod = pd.concat([feedbackDB,dfAux], ignore_index=True)
    return feedbackDBMod

def guardarComponentes(problemaID,componenteDB,representacion, Eval, Nb, Perturb, SampleSol, version):
    datosComponentes = {
        'ID_Problema': problemaID,
        'Representacion': representacion,
        'Evaluacion': Eval,
        'Vecindad': Nb,
        'Perturbacion': Perturb,
        'SolucionPrueba': SampleSol,
        'Version': version
    }
    dfAux = pd.DataFrame([datosComponentes])
    componenteDBMod = pd.concat([componenteDB,dfAux], ignore_index=True)
    return componenteDBMod

def guardarDatos(datos, header, path):
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

def cargarInspiraciones(inspirationDB: pd.DataFrame, seed = 42):
    #igual que componentes. Pero aqui se extrae del DF. Extraermos columnas 'Evaluacion' 'Vecindad' y 'Perturbacion'. Luego seleccionamos aleatoriamente usando la semilla definida al principio. No es necesario ejecutar el codigo. Solo se carga el texto y se le pasa eso. Si no se encuentra nada. Se returna NA
    Eval = "NA" 
    Perturb = "NA" 
    Nb = "NA"
    if not inspirationDB.empty:
        Eval = inspirationDB["Evaluacion"].sample(random_state=seed).iloc[0]
        Perturb = inspirationDB["Perturbacion"].sample(random_state=seed).iloc[0]
        Nb = inspirationDB["Vecindad"].sample(random_state=seed).iloc[0]
    return Eval, Perturb, Nb

def generarDiagnostico(context_message):
        tipo, error, tb  = sys.exc_info()
        
        causa = ""
        if error.__cause__:
            causa = f"\nCaused by: {type(error.__cause__).__name__}: {error.__cause__}"
        elif error.__context__ and not error.__suppress_context__:
            causa = f"\nDuring handling of: {type(error.__context__).__name__}: {error.__context__}"

        ultimoFrame = traceback.extract_tb(tb)[-1]
        linea = ultimoFrame.line
        lineno  = ultimoFrame.lineno
        if not linea:
            filename = ultimoFrame.filename
            if filename == "<string>":
            # linecache stores exec()'d source if you register it first (see below)
                linea = linecache.getline(filename, lineno).strip()
        if not linea:
            try:
                frame_obj = tb
                while frame_obj.tb_next:
                    frame_obj = frame_obj.tb_next
                source_lines, start = inspect.getsourcelines(frame_obj.tb_frame)
                offset = lineno - start
                if 0 <= offset < len(source_lines):
                    linea = source_lines[offset].strip()
            except (OSError, TypeError):
                pass

        if not linea:
            linea = "<source unavailable>"
        
        full_tb = "".join(traceback.format_tb(tb))
        return (f"{context_message}\n"
                f"Type: {tipo.__name__}\n"
                f"Message: {error}{causa}\n"
                f"Line; {lineno}\n"
                f"Code: {linea}"
                f"{'-'*60}\n"
                f"  Full Traceback:\n{full_tb}"
                f"{'='*60}")

evaluador = partial(cronometrarFuncion, evaluarComponentes)
evaluadorSP = partial(cronometrarFuncion, evaluarComponentesSP)

#Utiliza las DB para guardar las respuestas con versionado. Eso significa que hay que acceder a los archivos y escribir sobre ellos. 
#Podemos pasar el problema definido en la fase 1 directamente, en vez de sacar uno de la DB. Eso nos deja con dos variantes: seed, y problema predefinido. En ambos casos resultDB, feedbackDB y componenteDB son necesarios
#actualmente solo esta evaluando 1 componente a la vez, tiene que evaluar sets de componentes. Representacion, Vecindad y Evaluacion

def optimizarProblemaPreparadoConInspiraciones(problema,schema,problemData, componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame,inspiracionDB:pd.DataFrame, iteraciones: int, semilla: int):
        ### generacion inicial
        llms = generador()
        llms.cargarLLMs()
        problemaID, jsonProblema, knownSol, knownObj = extraerDatosProblema(problema)
        print(problemaID)
        objetivo, restricciones = jsonProblema['OBJECTIVE'], jsonProblema['CONSTRAINTS']
        inspiracionEval,inspiracionPerturb,inspiracionNB = cargarInspiraciones(inspiracionDB, semilla)
        Eval, Nb, Perturb, representacion, EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal = generacionComponentesInicial(schema, objetivo, restricciones,llms, knownSol, inspiracionEval,inspiracionPerturb,inspiracionNB)

        componenteDB = guardarComponentes(problemaID,componenteDB,representacion, Eval, Nb, Perturb, knownSol, iteraciones)
        i: int = 0
        
        ### proceso iterativo
        while i < iteraciones:
            resultados = []
            for heuristica in heuristicas:
                funcion, params, nombre, nombreCompleto = heuristica
                resultado, tiempo = evaluador(funcion,Eval, Nb, Perturb, problemData, knownSol,params)
                resultados.append((resultado,tiempo,nombre,nombreCompleto))
            ### Corte temprano
            feedbackCorrectitud,optimo, correctos = componentesCorrectos(Eval, problemData, knownSol,knownObj,resultados)
            if optimo:
                guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Optimo", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados)
                inspiracionDB = guardarComponentes(problemaID,inspiracionDB,representacion, Eval, Nb, Perturb, knownSol, i)
                return componenteDB, feedbackDB, resultDB, inspiracionDB
            elif correctos:
                guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Correcto", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados)
            else:
                guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Incorrecto", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados)
            ### Retroalimentacion
            feedbackPromptEval, feedbackPromptNb, feedbackPromptPerturb = generateFeedbackPrompts(schema, objetivo, restricciones,Eval, Nb, Perturb, knownSol,feedbackCorrectitud, "NA", "NA", i,resultados) 
            feedbackEval,FEvalTokenInput,FEvalTokenOutput,FEvalTokenTotal = generarFeedback(llms,feedbackPromptEval)
            feedbackNb,FNbTokenInput,FbTokenOutput,FNbTokenTotal = generarFeedback(llms,feedbackPromptNb)
            feedbackPerturb,FPerturbTokenInput,FPerturbTokenOutput,FPerturbTokenTotal = generarFeedback(llms,feedbackPromptPerturb)
            feedbackDB = guardarFeedback(problemaID,feedbackDB,representacion,Eval, Nb, Perturb, i,feedbackEval, feedbackNb, feedbackPerturb,FEvalTokenInput,FEvalTokenOutput,FEvalTokenTotal,FNbTokenInput,FbTokenOutput,FNbTokenTotal,FPerturbTokenInput,FPerturbTokenOutput,FPerturbTokenTotal) 
            Eval, Nb, Perturb, representacion, EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal = generacionNuevosComponentes(schema, objetivo, restricciones,llms,knownSol,feedbackEval,feedbackNb,feedbackPerturb,inspiracionEval,inspiracionPerturb,inspiracionNB)
            componenteDB = guardarComponentes(problemaID,componenteDB,representacion, Eval, Nb, Perturb, knownSol, i)
            i = i + 1
        resultados = []
        for heuristica in heuristicas:
            funcion, params, nombre = heuristica
            resultado, tiempo = evaluador(funcion,Eval, Nb, Perturb, problemData, knownSol,params)
            resultados.append((resultado,tiempo,nombre))
        feedbackCorrectitud,optimo, correctos = componentesCorrectos(Eval, problemData, knownSol,knownObj,resultados)
        if optimo:
            guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Optimo", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados)
            inspiracionDB = guardarComponentes(problemaID,inspiracionDB,representacion, Eval, Nb, Perturb, knownSol, i)
        elif correctos:
            guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Correcto", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados)
        else:
            guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Incorrecto", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados) 
        return componenteDB, feedbackDB, resultDB, inspiracionDB

def optimizarProblemaPreparado(problema,schema,problemData, componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame,inspiracionDB:pd.DataFrame, iteraciones: int, semilla: int):
        ### generacion inicial
        llms = generador()
        llms.cargarLLMs()
        problemaID, jsonProblema, knownSol, knownObj = extraerDatosProblema(problema)
        print(problemaID)
        objetivo, restricciones = jsonProblema['OBJECTIVE'], jsonProblema['CONSTRAINTS']
        Eval, Nb, Perturb, representacion,EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal = generacionComponentesInicial(schema, objetivo, restricciones,llms, knownSol, "NA","NA","NA")
        componenteDB = guardarComponentes(problemaID,componenteDB,representacion, Eval, Nb, Perturb, knownSol, iteraciones)
        i: int = 0
        ### proceso iterativo
        while i < iteraciones:
            resultados = []
            for heuristica in heuristicas:
                funcion, params, nombre, nombreCompleto = heuristica
                resultado, tiempo = evaluador(funcion,Eval, Nb, Perturb, problemData, knownSol,params)
                resultados.append((resultado,tiempo,nombre,nombreCompleto))
            ### Corte temprano
            feedbackCorrectitud,optimo, correctos = componentesCorrectos(Eval, problemData, knownSol,knownObj,resultados)
            if optimo:
                guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Optimo", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados)
                inspiracionDB = guardarComponentes(problemaID,inspiracionDB,representacion, Eval, Nb, Perturb, knownSol, i)
                return componenteDB, feedbackDB, resultDB, inspiracionDB
            elif correctos:
                guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Correcto", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados)
            else:
                guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Incorrecto", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados)
            ### Retroalimentacion
            feedbackPromptEval, feedbackPromptNb, feedbackPromptPerturb = generateFeedbackPrompts(schema, objetivo, restricciones,Eval, Nb, Perturb, knownSol,feedbackCorrectitud, "NA", "NA", i,resultados) 
            feedbackEval,FEvalTokenInput,FEvalTokenOutput,FEvalTokenTotal = generarFeedback(llms,feedbackPromptEval)
            feedbackNb,FNbTokenInput,FbTokenOutput,FNbTokenTotal = generarFeedback(llms,feedbackPromptNb)
            feedbackPerturb,FPerturbTokenInput,FPerturbTokenOutput,FPerturbTokenTotal = generarFeedback(llms,feedbackPromptPerturb)
            feedbackDB = guardarFeedback(problemaID,feedbackDB,representacion,Eval, Nb, Perturb, i,feedbackEval, feedbackNb, feedbackPerturb,FEvalTokenInput,FEvalTokenOutput,FEvalTokenTotal,FNbTokenInput,FbTokenOutput,FNbTokenTotal,FPerturbTokenInput,FPerturbTokenOutput,FPerturbTokenTotal) 
            Eval, Nb, Perturb, representacion, EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal = generacionNuevosComponentes(schema, objetivo, restricciones,llms,knownSol,feedbackEval,feedbackNb,feedbackPerturb,"NA","NA","NA")
            componenteDB = guardarComponentes(problemaID,componenteDB,representacion, Eval, Nb, Perturb, knownSol, i)
            i = i + 1
        resultados = []
        for heuristica in heuristicas:
            funcion, params, nombre = heuristica
            resultado, tiempo = evaluador(funcion,Eval, Nb, Perturb, problemData, knownSol,params)
            resultados.append((resultado,tiempo,nombre))
        feedbackCorrectitud,optimo, correctos = componentesCorrectos(Eval, problemData, knownSol,knownObj,resultados)
        if optimo:
            guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Optimo", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados)
            inspiracionDB = guardarComponentes(problemaID,inspiracionDB,representacion, Eval, Nb, Perturb, knownSol, i)
        elif correctos:
            guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Correcto", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados)
        else:
            guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Incorrecto", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados) 
        return componenteDB, feedbackDB, resultDB, inspiracionDB

## No usa sistema de preparacion. Requiere modificacion para trabajar sin preparacion. Requiere que se cargen los datos de la instancia directamente para trabajar 
#To-Do: Modificar para que no use problemdata quizas?. En este caso es literalmente el texto del problema. Problemdata se esta pasando como la solucion en vez de known sol.
def optimizarProblemaSinPreparar(problemaID, problema, knownSol, knownObj, componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame, iteraciones):
        llms = generador()
        llms.cargarLLMs()
        print(problemaID)
        Eval, Nb, Perturb, representacion, EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal = generacionComponentesInicialSP(problema,llms, knownSol, "NA","NA","NA") #Modificar aqui
        componenteDB = guardarComponentes(problemaID,componenteDB,representacion, Eval, Nb, Perturb, knownSol, iteraciones)
        i: int = 0
        ### proceso iterativo
        while i < iteraciones:
            resultados = []
            for heuristica in heuristicas:
                funcion, params, nombre, nombreCompleto = heuristica
                resultado, tiempo = evaluadorSP(funcion,Eval, Nb, Perturb, knownSol,params)
                resultados.append((resultado,tiempo,nombre,nombreCompleto))
            
            ##Corte Temprano
            feedbackCorrectitud,optimo, correctos = componentesCorrectosSP(Eval, knownSol,knownObj,resultados)
            if optimo:
                guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Optimo", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados)
                inspiracionDB = guardarComponentes(problemaID,inspiracionDB,representacion, Eval, Nb, Perturb, knownSol, i)
                return componenteDB, feedbackDB, resultDB, inspiracionDB
            elif correctos:
                guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Correcto", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados)
            else:
                guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Incorrecto", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados)

            ### Retroalimentacion
            feedbackPromptEval, feedbackPromptNb, feedbackPromptPerturb = generateFeedbackPromptSP(problema,Eval, Nb, Perturb, knownSol,feedbackCorrectitud, "NA", "NA", i,resultados) 
            feedbackEval,FEvalTokenInput,FEvalTokenOutput,FEvalTokenTotal = generarFeedback(llms,feedbackPromptEval)
            feedbackNb,FNbTokenInput,FbTokenOutput,FNbTokenTotal = generarFeedback(llms,feedbackPromptNb)
            feedbackPerturb,FPerturbTokenInput,FPerturbTokenOutput,FPerturbTokenTotal = generarFeedback(llms,feedbackPromptPerturb)
            feedbackDB = guardarFeedback(problemaID,feedbackDB,representacion,Eval, Nb, Perturb, i,feedbackEval, feedbackNb, feedbackPerturb,FEvalTokenInput,FEvalTokenOutput,FEvalTokenTotal,FNbTokenInput,FbTokenOutput,FNbTokenTotal,FPerturbTokenInput,FPerturbTokenOutput,FPerturbTokenTotal) 
            Eval, Nb, Perturb, representacion, EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal = generacionNuevosComponentesSP(problema,llms,knownSol,feedbackEval, feedbackNb, feedbackPerturb, "NA","NA","NA") 
            componenteDB = guardarComponentes(problemaID,componenteDB,representacion, Eval, Nb, Perturb, knownSol, i)
            i = i + 1
        resultados = []
        for heuristica in heuristicas:
            funcion, params, nombre, nombreCompleto = heuristica
            resultado, tiempo = evaluadorSP(funcion,Eval, Nb, Perturb, knownSol,params)
            resultados.append((resultado,tiempo,nombre,nombreCompleto))
        feedbackCorrectitud,optimo, correctos = componentesCorrectosSP(Eval, knownSol,knownObj,resultados)
        if optimo:
            guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Optimo", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados)
            inspiracionDB = guardarComponentes(problemaID,inspiracionDB,representacion, Eval, Nb, Perturb, knownSol, i)
        elif correctos:
            guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Correcto", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados)
        else:
            guardarResultados(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,knownSol, "Incorrecto", EvalTokenInput,EvalTokenOutput,EvalTokenTotal,NbTokenInput,NbTokenOutput,NbTokenTotal,PerturbTokenInput,PerturbTokenOutput,PerturbTokenTotal,resultados)       
        return componenteDB, feedbackDB, resultDB, inspiracionDB
