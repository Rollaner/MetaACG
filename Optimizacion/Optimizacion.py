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
    #print(Eval['REPRESENTATION'],Eval['EVAL_CODE'], Nb['NB_CODE'], Perturb['PERTURB_CODE'])
    return Eval, Nb, Perturb, Eval['REPRESENTATION']

def generacionNuevosComponentes(schema,objetivo, restricciones, llms, sampleSol,resultadoSA, resultadoILS,resultadoTS,feedbackTexto,inspiracionEval,inspiracionPerturb,inspiracionNB) -> Tuple[Any,Any,Any]: ## Requiere TLC para que considere todos los resultados. Mezclemolos en uno. 
    #Hay que hacer que revisa con cada tipo de resultado. No solo Simmulated Annealing
    promptEval = updateEvalPrompt(schema,objetivo, restricciones, resultadoTS, feedbackTexto, sampleSol,inspiracionEval)
    promptNb = updateNBPrompt(schema,objetivo, restricciones,resultadoTS, feedbackTexto, sampleSol,inspiracionNB)
    promptPerturb = updatePerturbPrompt(schema,objetivo, restricciones,resultadoTS, feedbackTexto, sampleSol,inspiracionPerturb)
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
    #print(Eval['REPRESENTATION'],Eval['EVAL_CODE'], Nb['NB_CODE'], Perturb['PERTURB_CODE'])
    return Eval, Nb, Perturb,Eval['REPRESENTATION']

#Utiliza las DB para guardar las respuestas con versionado. Eso significa que hay que acceder a los archivos y escribir sobre ellos. 
#Podemos pasar el problema definido en la fase 1 directamente, en vez de sacar uno de la DB. Eso nos deja con dos variantes: seed, y problema predefinido. En ambos casos resultDB, feedbackDB y componenteDB son necesarios
#actualmente solo esta evaluando 1 componente a la vez, tiene que evaluar sets de componentes. Representacion, Vecindad y Evaluacion


def optimizarProblemaPreparadoDB(problema,schema,problemData, componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame,inspiracionDB:pd.DataFrame, iteraciones: int, semilla: int):
        ### generacion inicial
        llms = generador()
        llms.cargarLLMs()
        problemaID, jsonProblema, knownSol, knownObj = extraerDatosProblema(problema)
        print(problemaID)
        objetivo, restricciones = jsonProblema['OBJECTIVE'], jsonProblema['CONSTRAINTS']
        inspiracionEval,inspiracionPerturb,inspiracionNB = cargarInspiraciones(inspiracionDB, semilla)
        Eval, Nb, Perturb, representacion = generacionComponentesInicial(schema, objetivo, restricciones,llms, knownSol, inspiracionEval,inspiracionPerturb,inspiracionNB)
        componenteDB = guardarComponentes(problemaID,componenteDB,representacion, Eval, Nb, Perturb, knownSol, iteraciones)
        respuestas = [] #Temporal, para guardar todas las respuestas de generacion de componentes para tener la metadata.
        feedbacks = []
        i = 0
        ### proceso iterativo
        while i < iteraciones:
            resultadoSA, tiempoSA = cronometrarFuncion(evaluarComponentes,Heuristicas.SimulatedAnnealing.SA,Eval, Nb, Perturb,problemData, knownSol,[1000,10,0.9])
            resultadoILS, tiempoILS = cronometrarFuncion(evaluarComponentes,Heuristicas.IteratedLocalSearch.ILS, Eval, Nb, Perturb,problemData, knownSol,[200,0.1])
            resultadoTS, tiempoTS = cronometrarFuncion(evaluarComponentes,Heuristicas.TabooSearch.TS, Eval, Nb, Perturb,problemData, knownSol,[200,10,7])
            resultadoHC, tiempoHC = cronometrarFuncion(evaluarComponentes,Heuristicas.HillClimbing.HC, Eval, Nb, Perturb,problemData, knownSol,[200]) 
            print(resultadoSA)
            print(resultadoILS)
            print(resultadoTS)
            print(resultadoHC)
            resultDB = guardarResultado(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,resultadoSA,knownSol, "NA", "SA", tiempoSA)
            resultDB = guardarResultado(problemaID,i,resultDB,representacion,Eval,Nb,Perturb,resultadoILS,knownSol, "NA", "ILS", tiempoILS)
            resultDB = guardarResultado(problemaID,i,resultDB,representacion,Eval,Nb,Perturb,resultadoTS,knownSol, "NA", "TS", tiempoTS)
            resultDB = guardarResultado(problemaID,i,resultDB,representacion,Eval,Nb,Perturb,resultadoHC,knownSol, "NA", "HC", tiempoHC)
            #bloque de pruebas usando dataset.
            #for sample in samples:
            #    problemData = convertidor(sample)
            #    resultadoSA, tiempoSA = cronometrarFuncion(evaluarComponentes,Heuristicas.SimulatedAnnealing.SA,Eval, Nb, Perturb,problemData, SampleSol,[1000,10,0.9])
            #    resultadoILS, tiempoILS = cronometrarFuncion(evaluarComponentes,Heuristicas.IteratedLocalSearch.ILS, Eval, Nb, Perturb,problemData, SampleSol,[44,0.1])
            #    resultadoTS, tiempoTS = cronometrarFuncion(evaluarComponentes,Heuristicas.TabooSearch.TS, Eval, Nb, Perturb,problemData, SampleSol,[44,10,7])
            #    resultadoHC, tiempoHC = cronometrarFuncion(evaluarComponentes,Heuristicas.HillClimbing.HC, Eval, Nb, Perturb,problemData, SampleSol,44) 
            #    resultDB = guardarResultado(problemaID,resultDB,representacion, Eval,Nb,Perturb,resultadoSA,SampleSol, sample.solution, "SA", tiempoSA)
            #    resultDB = guardarResultado(problemaID,resultDB,representacion,Eval,Nb,Perturb,resultadoILS,SampleSol, sample.solution, "ILS", tiempoILS)
            #    resultDB = guardarResultado(problemaID,resultDB,representacion,Eval,Nb,Perturb,resultadoTS,SampleSol, sample.solution, "TS", tiempoTS)
            #    resultDB = guardarResultado(problemaID,resultDB,representacion,Eval,Nb,Perturb,resultadoHC,SampleSol, sample.solution, "HC", tiempoHC)
            ### Corte temprano

            if componentesCorrectos(knownSol,knownObj,resultadoSA,resultadoTS,resultadoILS,resultadoHC):
                inspiracionDB = guardarComponentes(problemaID,inspiracionDB,representacion, Eval, Nb, Perturb, knownSol, i)
                return componenteDB, feedbackDB, resultDB, inspiracionDB
            ### Retroalimentacion
            feedbackPrompt = generateFeedbackPrompt(problemData, objetivo, restricciones,Eval, Nb, Perturb, knownSol,resultadoSA, resultadoILS,resultadoTS, "NA", "NA", i) #Necesita trabajar con el nuevo sistema de JSON
            feedback = llms.generarFeedback(feedbackPrompt) 
            feedbacks.append(feedback)
            feedbackTexto = feedback.content[0]['text']
            print(f"feedback:{'-'*60}\n")
            print(feedbackTexto)
            feedbackDB = guardarFeedback(problemaID,feedbackDB,representacion,Eval, Nb, Perturb, i,feedbackTexto)
            respuesta = generacionNuevosComponentes(schema, objetivo, restricciones,llms,knownSol,resultadoSA,resultadoILS, resultadoTS,feedbackTexto,inspiracionEval,inspiracionPerturb,inspiracionNB)
            respuestas.append(respuesta)
            Eval, Nb, Perturb, representacion = respuesta
            componenteDB = guardarComponentes(problemaID,componenteDB,representacion, Eval, Nb, Perturb, knownSol, i)
            i = i + 1
        resultadoSA, tiempoSA = cronometrarFuncion(evaluarComponentes,Heuristicas.SimulatedAnnealing.SA,Eval, Nb, Perturb,problemData,knownSol,[1000,10,0.9])
        resultadoILS, tiempoILS = cronometrarFuncion(evaluarComponentes,Heuristicas.IteratedLocalSearch.ILS, Eval, Nb, Perturb,problemData, knownSol,[200,0.1])
        resultadoTS, tiempoTS = cronometrarFuncion(evaluarComponentes,Heuristicas.TabooSearch.TS, Eval, Nb, Perturb,problemData, knownSol,[200,10,7])
        resultadoHC, tiempoHC = cronometrarFuncion(evaluarComponentes,Heuristicas.HillClimbing.HC, Eval, Nb, Perturb,problemData, knownSol,[200]) 
        print(resultadoSA)
        print(resultadoILS)
        print(resultadoTS)
        resultDB = guardarResultado(problemaID,i,resultDB,representacion, Eval,Nb,Perturb,resultadoSA,knownSol, "NA", "SA", tiempoSA)
        resultDB = guardarResultado(problemaID,i,resultDB,representacion,Eval,Nb,Perturb,resultadoILS,knownSol, "NA", "ILS", tiempoILS)
        resultDB = guardarResultado(problemaID,i,resultDB,representacion,Eval,Nb,Perturb,resultadoTS,knownSol, "NA", "TS", tiempoTS)
        resultDB = guardarResultado(problemaID,i,resultDB,representacion,Eval,Nb,Perturb,resultadoHC,knownSol, "NA", "HC", tiempoHC)       
        return componenteDB, feedbackDB, resultDB, inspiracionDB

## Detecta si todo obtuvo resultados validos, y si alguna de las metaheuristicas llego al optimo. Si una lo hizo, sabemos que se puede, y que los componententes estan correctos.
def componentesCorrectos(knownSol, knownObj, *args):
    optimoPresente:bool = False
    errorNoDetectado:bool = True
    for resultados in args:
        try: 
            currentSolucion,currentScore, mejorSolucion, mejorScore = resultados
        except Exception as e:
            #print(f"Error fatal de procesado en: {resultados}")
            return False #estos NO estan correctos, algo no paso.
        if (mejorSolucion == knownSol and mejorScore == knownObj):
            optimoPresente = True
        if (not isinstance(currentSolucion, (list, tuple)) or not isinstance(currentScore, numbers.Real) or not isinstance(mejorSolucion, (list, tuple))or not isinstance(mejorScore, numbers.Real)):
            errorNoDetectado = False        
    return (optimoPresente and errorNoDetectado)


## No usa sistema de preparacion, Deprecado
def optimizarProblemaSinPreparar(problemaID, problema, problemData, SampleSol,componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame, iteraciones):
        ### generacion inicial
        llms = generador()
        llms.cargarLLMs()
        print(problemaID, "NA", "NA")
        Eval, Nb, Perturb, representacion = generacionComponentesInicial(problemaID, problema, SampleSol)
        componenteDB = guardarComponentes(problemaID,componenteDB,representacion, Eval, Nb, Perturb, SampleSol, i)
        respuestas = [] #Temporal, para guardar todas las respuestas de generacion de componentes para tener la metadata.
        feedbacks = []
        i = 0

        ### proceso iterativo
        while i < iteraciones:
            resultadoSA, tiempoSA = cronometrarFuncion(evaluarComponentes,Heuristicas.SimulatedAnnealing.SA,Eval, Nb, Perturb,problemData, SampleSol,[1000,10,0.9])
            resultadoILS, tiempoILS = cronometrarFuncion(evaluarComponentes,Heuristicas.IteratedLocalSearch.ILS, Eval, Nb, Perturb,problemData, SampleSol,[44,0.1])
            resultadoTS, tiempoTS = cronometrarFuncion(evaluarComponentes,Heuristicas.TabooSearch.TS, Eval, Nb, Perturb,problemData, SampleSol,[44,10,7])
            print(resultadoSA)
            print(resultadoILS)
            print(resultadoTS)
            resultDB = guardarResultado(problemaID,resultDB,representacion, Eval,Nb,Perturb,resultadoSA,SampleSol, "NA", "SA", tiempoSA)
            resultDB = guardarResultado(problemaID,resultDB,representacion,Eval,Nb,Perturb,resultadoILS,SampleSol, "NA", "ILS", tiempoILS)
            resultDB = guardarResultado(problemaID,resultDB,representacion,Eval,Nb,Perturb,resultadoTS,SampleSol, "NA", "TS", tiempoTS)
            feedbackPrompt = generateFeedbackPrompt(problema,Eval, Nb, Perturb, SampleSol,resultadoSA, resultadoILS,resultadoTS, "NA", "NA") #Necesita trabajar con el nuevo sistema de JSON
            feedback = llms.generarFeedback(feedbackPrompt) 
            feedbacks.append(feedback)
            feedbackTexto = feedback.content[0]['text']
            print(feedbackTexto)
            feedbackDB = guardarFeedback(problemaID,feedbackDB,representacion,Eval, Nb, Perturb, i,feedbackTexto)
            
            ### Retroalimentacion
            respuesta = generacionNuevosComponentes(problemaID,problema,llms,SampleSol,resultadoSA,resultadoILS, resultadoTS,feedbackTexto)
            respuestas.append(respuesta)
            Eval, Nb, Perturb, representacion = respuesta
            componenteDB = guardarComponentes(problemaID,componenteDB,representacion, Eval, Nb, Perturb, SampleSol, i)
            i = i + 1
            if i % 3 == 0:
                llms = reiniciarLLMS()
                print("Maquinas re-instanciadas")
        resultadoSA, tiempoSA = cronometrarFuncion(evaluarComponentes,Heuristicas.SimulatedAnnealing.SA,Eval, Nb, Perturb,problemData, SampleSol,[1000,10,0.9])
        resultadoILS, tiempoILS = cronometrarFuncion(evaluarComponentes,Heuristicas.IteratedLocalSearch.ILS, Eval, Nb, Perturb,problemData, SampleSol,[44,0.1])
        resultadoTS, tiempoTS = cronometrarFuncion(evaluarComponentes,Heuristicas.TabooSearch.TS, Eval, Nb, Perturb,problemData, SampleSol,[44,10,7])
        print(resultadoSA)
        print(resultadoILS)
        print(resultadoTS)
        resultDB = guardarResultado(problemaID,resultDB,representacion, Eval,Nb,Perturb,resultadoSA,SampleSol, "NA", "SA", tiempoSA)
        resultDB = guardarResultado(problemaID,resultDB,representacion,Eval,Nb,Perturb,resultadoILS,SampleSol, "NA", "ILS", tiempoILS)
        resultDB = guardarResultado(problemaID,resultDB,representacion,Eval,Nb,Perturb,resultadoTS,SampleSol, "NA", "TS", tiempoTS)     
        return componenteDB, feedbackDB, resultDB

def cargarDatosProblema(problemData, standardDataclass, convertidorText):
    convertidor = cargarComponente(convertidorText)
    problemaActual = convertidor(problemData,standardDataclass)
    return problemaActual

def evaluarComponentes(Heuristica,Eval, Nb, Perturb,problemData, SampleSol, Params):
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

# No prueban la soluciuon con el algoritmo de evaluacion a priori. Se lo pasan al solver directamente. Menos trazabilidad
def evaluarComponentesSAud(Eval, Nb, Perturb,SampleSol):
    #Guardar excepciones como string, para retornarlas como debug
    try:
        evaluacion = cargarComponente(Eval['EVAL_CODE'])
        print(f"Loaded 'EVAL_CODE' into variable 'evaluate_solution'. Name: {eval.__name__}")
    except Exception as e:
        return(f"Failed to load EVAL_CODE: {e}")
    try:
        vecindad = cargarComponente(Nb['NB_CODE'])
        print(f"Loaded 'NB_CODE' into variable 'generate_neighbour'. Name: {vecindad.__name__}")
    except Exception as e:
        return(f"Failed to load NB_CODE: {e}")
    try:
        perturb = cargarComponente(Perturb['PERTURB_CODE'])
        print(f"Loaded 'PERTURB_CODE' into variable 'perturb_solution'. Name: {perturb.__name__}")
    except Exception as e:
        return(f"Failed to load PERTURB_CODE: {e}")
    try:
        solucionPrueba = SampleSol
    except Exception as e:
        try:
            solucionPrueba = ast.literal_eval(solucionPrueba.replace(' ', ''))
        except Exception as e:
            return(f"Failed to load SAMPLE_SOL: {e}")
    
    try:
        resultadoSA = Heuristicas.SimulatedAnnealing.SA(solucionPrueba,solucionPrueba,evaluacion(solucionPrueba),vecindad,evaluacion,1000,10,0.9)
    except Exception as e:
        return(f"Failed to run target heuristic: {e}.  Signature def SA(solution,best_sol, best_score, generate_neighbour(), evaluate_solution(), TEMP, MIN_TEMP, cooling_factor)")
    #Cargar heuristicas, retornar resultados de cada una.
    return resultadoSA

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


def guardarResultado(problemaID, iteracion,resultDB, representacion, evaluacion, vecindad, perturbacion, resultados,mejorSolucion, optimo, MH, tiempo):
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
        'Tiempo': tiempo
    }
    dfAux = pd.DataFrame([datosResultado])
    dfAux = dfAux.dropna(axis=1, how='all')
    resultDBMod = pd.concat([resultDB,dfAux], ignore_index=True) #
    return resultDBMod

def guardarFeedback(problemaID,feedbackDB, representacion, Eval, Nb, Perturb, version, feedback):
    datosFeedback = {
        'ID_Problema': problemaID,
        'Representacion': representacion,
        'Evaluacion': Eval,
        'Vecindad': Nb,
        'Perturbacion': Perturb,
        'Version': version,
        'Feedback': feedback
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