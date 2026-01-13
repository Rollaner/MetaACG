import ast
import time
from typing import Any, Callable, Dict, Tuple
from Instancias import Instancia,DataLoader
from Generador import generador
from .PromptSamplerOP import *
import Heuristicas.IteratedLocalSearch
import Heuristicas.SimulatedAnnealing
import Heuristicas.TabooSearch
import traceback
import sys
import json

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
    if isinstance(problemaSample, pd.DataFrame):
        problemaID = problemaSample.Instancia.iloc[0]
        inspiraciones=json.loads(problemaSample.Respuesta.iloc[0])
        knownSol = problemaSample['Resultado esperado'].iloc[0]
        knownObj = problemaSample['Valor Objetivo'].iloc[0]
    else:
        problemaID = problemaSample.Instancia
        inspiraciones=json.loads(problemaSample.Respuesta)
        knownSol = problemaSample[7] #Resultado_esperado
        knownObj = problemaSample[8] #Valor_Objetivo
    return problemaID, inspiraciones, knownSol, knownObj

def generacionComponentesInicial(problemaID,defProblema, llms, sampleSol):
    promptEval = generateEvalPrompt(problemaID,defProblema, sampleSol)
    promptNb = generateNBPrompt(problemaID,defProblema, sampleSol)
    promptPerturb = generatePerturbPrompt(problemaID,defProblema, sampleSol)
    RawEval = llms.generarComponente(promptEval)
    RawNb = llms.generarComponente(promptNb)
    RawPerturb = llms.generarComponente(promptPerturb)
    textoBruto = RawEval.content[0]['text']
    Eval = json.loads(textoBruto)
    textoBruto = RawNb.content[0]['text']
    Nb = json.loads(textoBruto)
    textoBruto = RawPerturb.content[0]['text']
    Perturb = json.loads(textoBruto)
    print(Eval['REPRESENTATION'],Eval['EVAL_CODE'], Nb['NB_CODE'], Perturb['PERTURB_CODE'])
    return Eval, Nb, Perturb, Eval['REPRESENTATION']

def generacionNuevosComponentes(problemaID,defProblema, llms, sampleSol,resultadoSA, resultadoILS,resultadoTS,feedbackTexto) -> Tuple[Any,Any,Any]:
    #Hay que hacer que revisa con cada tipo de resultado. No solo Simmulated Annealing
    promptEval = updateEvalPrompt(problemaID,defProblema, resultadoSA, feedbackTexto, sampleSol)
    promptNb = updateNBPrompt(problemaID,defProblema,resultadoSA, feedbackTexto, sampleSol)
    promptPerturb = updatePerturbPrompt(problemaID,defProblema,resultadoSA, feedbackTexto, sampleSol)
    RawEval = llms.generarComponente(promptEval)
    RawNb = llms.generarComponente(promptNb)
    RawPerturb = llms.generarComponente(promptPerturb)
    textoBruto = RawEval.content[0]['text']
    Eval = json.loads(textoBruto)
    textoBruto = RawNb.content[0]['text']
    Nb = json.loads(textoBruto)
    textoBruto = RawPerturb.content[0]['text']
    Perturb = json.loads(textoBruto)
    print(Eval['REPRESENTATION'],Eval['EVAL_CODE'], Nb['NB_CODE'], Perturb['PERTURB_CODE'])
    return Eval, Nb, Perturb,Eval['REPRESENTATION']

#Utiliza las DB para guardar las respuestas con versionado. Eso significa que hay que acceder a los archivos y escribir sobre ellos. 
#Podemos pasar el problema definido en la fase 1 directamente, en vez de sacar uno de la DB. Eso nos deja con dos variantes: seed, y problema predefinido. En ambos casos resultDB, feedbackDB y componenteDB son necesarios
#actualmente solo esta evaluando 1 componente a la vez, tiene que evaluar sets de componentes. Representacion, Vecindad y Evaluacion
def optimizarProblemaAleatorio(problemaDB:pd.DataFrame,componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame,semilla, iteraciones):
        rawProblema = sampleProblemaDB(problemaDB,semilla)
        componenteDB, feedbackDB, resultDB = optimizarProblemaPreparado(rawProblema, componenteDB, resultDB, feedbackDB, iteraciones)
        return componenteDB, feedbackDB, resultDB

def optimizarProblemaAleatorioH(problemaDB:pd.DataFrame,componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame,semilla, iteraciones):
        rawProblema = sampleProblemaDB(problemaDB,semilla)
        componenteDB, feedbackDB, resultDB = optimizarProblemaPreparado(rawProblema, componenteDB, resultDB, feedbackDB, iteraciones)
        return componenteDB, feedbackDB, resultDB

## Idem que el anterior, pero no tiene acceso a los datos de la solucion
def optimizarProblemaPreparado(problemaID, problema, sampleSol,componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame, iteraciones):
        ### generacion inicial
        llms = generador()
        llms.cargarLLMs()
        print(problemaID, "NA", "NA")
        Eval, Nb, Perturb, representacion = generacionComponentesInicial(problemaID, problema,llms, sampleSol)
        componenteDB = guardarComponentes(problemaID,componenteDB,representacion, Eval, Nb, Perturb, sampleSol, i)
        respuestas = [] #Temporal, para guardar todas las respuestas de generacion de componentes para tener la metadata.
        feedbacks = []
        i = 0

        ### proceso iterativo
        while i < iteraciones:
            resultadoSA, tiempoSA = cronometrarFuncion(evaluarComponentesSA, Eval, Nb, Perturb, sampleSol)
            resultadoILS, tiempoILS = cronometrarFuncion(evaluarComponentesILS, Eval, Nb, Perturb, sampleSol)
            resultadoTS, tiempoTS = cronometrarFuncion(evaluarComponentesTS, Eval, Nb, Perturb, sampleSol)
            print(resultadoSA)
            print(resultadoILS)
            print(resultadoTS)
            resultDB = guardarResultado(problemaID,resultDB,representacion, Eval,Nb,Perturb,resultadoSA,sampleSol, "NA", "SA", tiempoSA)
            resultDB = guardarResultado(problemaID,resultDB,representacion,Eval,Nb,Perturb,resultadoILS,sampleSol, "NA", "ILS", tiempoILS)
            resultDB = guardarResultado(problemaID,resultDB,representacion,Eval,Nb,Perturb,resultadoTS,sampleSol, "NA", "TS", tiempoTS)
            feedbackPrompt = generateFeedbackPrompt(problema,Eval, Nb, Perturb, sampleSol,resultadoSA, resultadoILS,resultadoTS, "NA", "NA") #Necesita trabajar con el nuevo sistema de JSON
            feedback = llms.generarFeedback(feedbackPrompt) 
            feedbacks.append(feedback)
            feedbackTexto = feedback.content[0]['text']
            print(feedbackTexto)
            feedbackDB = guardarFeedback(problemaID,feedbackDB,representacion,Eval, Nb, Perturb, i,feedbackTexto)
            
            ### Retroalimentacion
            respuesta = generacionNuevosComponentes(problemaID,problema,llms,sampleSol,resultadoSA,resultadoILS, resultadoTS,feedbackTexto)
            respuestas.append(respuesta)
            Eval, Nb, Perturb, representacion = respuesta
            componenteDB = guardarComponentes(problemaID,componenteDB,representacion, Eval, Nb, Perturb, sampleSol, i)
            i = i + 1
            if i % 3 == 0:
                llms = reiniciarLLMS()
                print("Maquinas re-instanciadas")
        resultadoSA, tiempoSA = cronometrarFuncion(evaluarComponentesSA,Eval, Nb, Perturb, sampleSol)
        resultadoILS, tiempoILS = cronometrarFuncion(evaluarComponentesILS, Eval, Nb, Perturb, sampleSol)
        resultadoTS, tiempoTS = cronometrarFuncion(evaluarComponentesTS, Eval, Nb, Perturb, sampleSol)
        print(resultadoSA)
        print(resultadoILS)
        print(resultadoTS)
        resultDB = guardarResultado(problemaID,resultDB,representacion, Eval,Nb,Perturb,resultadoSA,sampleSol, "NA", "SA", tiempoSA)
        resultDB = guardarResultado(problemaID,resultDB,representacion,Eval,Nb,Perturb,resultadoILS,sampleSol, "NA", "ILS", tiempoILS)
        resultDB = guardarResultado(problemaID,resultDB,representacion,Eval,Nb,Perturb,resultadoTS,sampleSol, "NA", "TS", tiempoTS)       
        return componenteDB, feedbackDB, resultDB


def optimizarProblemaPreparadoDB(problema,componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame, iteraciones):
        ### generacion inicial
        llms = generador()
        llms.cargarLLMs()
        problemaID, defProblema, SampleSol, _ = extraerDatosProblema(problema)
        print(problemaID, "NA", "NA")
        Eval, Nb, Perturb, representacion = generacionComponentesInicial(problemaID, defProblema,llms, SampleSol)
        componenteDB = guardarComponentes(problemaID,componenteDB,representacion, Eval, Nb, Perturb, SampleSol, iteraciones)
        respuestas = [] #Temporal, para guardar todas las respuestas de generacion de componentes para tener la metadata.
        feedbacks = []
        i = 0

        ### proceso iterativo
        while i < iteraciones:
            resultadoSA, tiempoSA = cronometrarFuncion(evaluarComponentesSA, Eval, Nb, Perturb, SampleSol)
            resultadoILS, tiempoILS = cronometrarFuncion(evaluarComponentesILS, Eval, Nb, Perturb, SampleSol)
            resultadoTS, tiempoTS = cronometrarFuncion(evaluarComponentesTS, Eval, Nb, Perturb, SampleSol)
            print(resultadoSA)
            print(resultadoILS)
            print(resultadoTS)
            resultDB = guardarResultado(problemaID,resultDB,representacion, Eval,Nb,Perturb,resultadoSA,SampleSol, "NA", "SA", tiempoSA)
            resultDB = guardarResultado(problemaID,resultDB,representacion,Eval,Nb,Perturb,resultadoILS,SampleSol, "NA", "ILS", tiempoILS)
            resultDB = guardarResultado(problemaID,resultDB,representacion,Eval,Nb,Perturb,resultadoTS,SampleSol, "NA", "TS", tiempoTS)
            feedbackPrompt = generateFeedbackPrompt(defProblema,Eval, Nb, Perturb, SampleSol,resultadoSA, resultadoILS,resultadoTS, "NA", "NA") #Necesita trabajar con el nuevo sistema de JSON
            feedback = llms.generarFeedback(feedbackPrompt) 
            feedbacks.append(feedback)
            feedbackTexto = feedback.content[0]['text']
            print(feedbackTexto)
            feedbackDB = guardarFeedback(problemaID,feedbackDB,representacion,Eval, Nb, Perturb, i,feedbackTexto)
            
            ### Retroalimentacion
            respuesta = generacionNuevosComponentes(problemaID,defProblema,llms,SampleSol,resultadoSA,resultadoILS, resultadoTS,feedbackTexto)
            respuestas.append(respuesta)
            Eval, Nb, Perturb, representacion = respuesta
            componenteDB = guardarComponentes(problemaID,componenteDB,representacion, Eval, Nb, Perturb, SampleSol, i)
            i = i + 1
            if i % 3 == 0:
                llms = reiniciarLLMS()
                print("Maquinas re-instanciadas")
        resultadoSA, tiempoSA = cronometrarFuncion(evaluarComponentes,Heuristicas.SimulatedAnnealing.SA,Eval, Nb, Perturb, SampleSol,[1000,10,0.9])
        resultadoILS, tiempoILS = cronometrarFuncion(evaluarComponentes,Heuristicas.IteratedLocalSearch.ILS, Eval, Nb, Perturb, SampleSol,[44,0.1])
        resultadoTS, tiempoTS = cronometrarFuncion(evaluarComponentes,Heuristicas.TabooSearch.TS, Eval, Nb, Perturb, SampleSol,[44,10,7])
        print(resultadoSA)
        print(resultadoILS)
        print(resultadoTS)
        resultDB = guardarResultado(problemaID,resultDB,representacion, Eval,Nb,Perturb,resultadoSA,SampleSol, "NA", "SA", tiempoSA)
        resultDB = guardarResultado(problemaID,resultDB,representacion,Eval,Nb,Perturb,resultadoILS,SampleSol, "NA", "ILS", tiempoILS)
        resultDB = guardarResultado(problemaID,resultDB,representacion,Eval,Nb,Perturb,resultadoTS,SampleSol, "NA", "TS", tiempoTS)       
        return componenteDB, feedbackDB, resultDB

## No usa sistema de preparacion
def optimizarProblemaSinPreparar(problemaID, problema, SampleSol,componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame, iteraciones):
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
            resultadoSA, tiempoSA = cronometrarFuncion(evaluarComponentesSA, Eval, Nb, Perturb, SampleSol)
            resultadoILS, tiempoILS = cronometrarFuncion(evaluarComponentesILS, Eval, Nb, Perturb, SampleSol)
            resultadoTS, tiempoTS = cronometrarFuncion(evaluarComponentesTS, Eval, Nb, Perturb, SampleSol)
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
        resultadoSA, tiempoSA = cronometrarFuncion(evaluarComponentesSA,Eval, Nb, Perturb, SampleSol)
        resultadoILS, tiempoILS = cronometrarFuncion(evaluarComponentesILS, Eval, Nb, Perturb, SampleSol)
        resultadoTS, tiempoTS = cronometrarFuncion(evaluarComponentesTS, Eval, Nb, Perturb, SampleSol)
        print(resultadoSA)
        print(resultadoILS)
        print(resultadoTS)
        resultDB = guardarResultado(problemaID,resultDB,representacion, Eval,Nb,Perturb,resultadoSA,SampleSol, "NA", "SA", tiempoSA)
        resultDB = guardarResultado(problemaID,resultDB,representacion,Eval,Nb,Perturb,resultadoILS,SampleSol, "NA", "ILS", tiempoILS)
        resultDB = guardarResultado(problemaID,resultDB,representacion,Eval,Nb,Perturb,resultadoTS,SampleSol, "NA", "TS", tiempoTS)     
        return componenteDB, feedbackDB, resultDB

def evaluarComponentes(Heuristica,Eval, Nb, Perturb, SampleSol, Params):
    #Guardar excepciones como string, para retornarlas como debug
    try:
        solucionPrueba = ast.literal_eval(SampleSol['SAMPLE_SOL'].strip().replace(' ', ''))
    except Exception as e:
        return(generarDiagnostico(f"Failed to load SAMPLE_SOL: {e}"))
    try:
        evaluacion = cargarComponente(Eval['EVAL_CODE'])
        print(f"Loaded 'EVAL_CODE' into variable 'evaluate_solution'. Name: {eval.__name__}")
    except Exception as e:
        return(generarDiagnostico(f"Failed to load EVAL_CODE: {e}"))
    try:
        vecindad = cargarComponente(Nb['NB_CODE'])
        print(f"Loaded 'NB_CODE' into variable 'generate_neighbour'. Name: {vecindad.__name__}")
    except Exception as e:
        return(generarDiagnostico(f"Failed to load NB_CODE: {e}"))
    try:
        perturb = cargarComponente(Perturb['PERTURB_CODE'])
        print(f"Loaded 'PERTURB_CODE' into variable 'perturb_solution'. Name: {perturb.__name__}")
    except Exception as e:
        return(generarDiagnostico(f"Failed to load PERTURB_CODE: {e}"))
    try:
        valor = evaluacion(solucionPrueba)
    except Exception as e:
        return(generarDiagnostico(f"Failed to evaluate SAMPLE_SOL with EVAL_CODE: {e}"))
    try:
        resultado = Heuristica(solucionPrueba,solucionPrueba,evaluacion(solucionPrueba),vecindad,perturb,evaluacion,*Params)
    except Exception as e:
        return(generarDiagnostico(f"Failed to run target heuristic: {e}.  Signature def SA(solution,best_sol, best_score, generate_neighbour(), evaluate_solution(), TEMP, MIN_TEMP, cooling_factor)"))
    #Cargar heuristicas, retornar resultados de cada una.
    return resultado

def evaluarComponentesILS(Eval, Nb, Perturb, SampleSol):
    #Guardar excepciones como string, para retornarlas como debug
    try:
        solucionPrueba = ast.literal_eval(SampleSol.strip().replace(' ', ''))
    except Exception as e:
        return(generarDiagnostico(f"Failed to load SAMPLE_SOL: {e}"))
    try:
        evaluacion = cargarComponente(Eval['EVAL_CODE'])
        print(f"Loaded 'EVAL_CODE' into variable 'evaluate_solution'. Name: {eval.__name__}")
    except Exception as e:
        return(generarDiagnostico(f"Failed to load EVAL_CODE: {e}"))
    try:
        vecindad = cargarComponente(Nb['NB_CODE'])
        print(f"Loaded 'NB_CODE' into variable 'generate_neighbour'. Name: {vecindad.__name__}")
    except Exception as e:
        return(generarDiagnostico(f"Failed to load NB_CODE: {e}"))
    try:
        perturb = cargarComponente(Perturb['PERTURB_CODE'])
        print(f"Loaded 'PERTURB_CODE' into variable 'perturb_solution'. Name: {perturb.__name__}")
    except Exception as e:
        return(generarDiagnostico(f"Failed to load PERTURB_CODE: {e}"))
    try:
        valor = evaluacion(solucionPrueba)
    except Exception as e:
        return(generarDiagnostico(f"Failed to evaluate SAMPLE_SOL with EVAL_CODE: {e}"))
    try:
        resultadoILS = Heuristicas.IteratedLocalSearch.ILS(solucionPrueba,solucionPrueba,evaluacion(solucionPrueba),vecindad,perturb,evaluacion,44,0.1)
    except Exception as e:
        return(generarDiagnostico(f"Failed to run target heuristic: {e}.  Signature def SA(solution,best_sol, best_score, generate_neighbour(), evaluate_solution(), TEMP, MIN_TEMP, cooling_factor)"))
    #Cargar heuristicas, retornar resultados de cada una.
    return resultadoILS


#Deprecado
def evaluarComponentesSA(Eval, Nb, Perturb, SampleSol):
    #Guardar excepciones como string, para retornarlas como debug
    try:
        solucionPrueba = ast.literal_eval(SampleSol['SAMPLE_SOL'].strip().replace(' ', ''))
    except Exception as e:
        return(f"Failed to load SAMPLE_SOL: {e}")
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
        valor = evaluacion(solucionPrueba)
    except Exception as e:
        return(f"Failed to evaluate SAMPLE_SOL with EVAL_CODE: {e}")
    try:
        resultadoSA = Heuristicas.SimulatedAnnealing.SA(solucionPrueba,solucionPrueba,valor,vecindad,evaluacion,1000,10,0.9)
    except Exception as e:
        return(f"Failed to run target heuristic: {e}.  Signature def SA(solution,best_sol, best_score, generate_neighbour(), evaluate_solution(), TEMP, MIN_TEMP, cooling_factor)")
    #Cargar heuristicas, retornar resultados de cada una.
    return resultadoSA
#Deprecado
def evaluarComponentesTS(Eval, Nb, Perturb, SampleSol):
    #Guardar excepciones como string, para retornarlas como debug
    try:
        solucionPrueba = ast.literal_eval(SampleSol['SAMPLE_SOL'].strip().replace(' ', ''))
    except Exception as e:
        return(f"Failed to load SAMPLE_SOL: {e}")
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
        valor = evaluacion(solucionPrueba)
    except Exception as e:
        return(f"Failed to evaluate SAMPLE_SOL with EVAL_CODE: {e}")
    try:
        resultadoTS = Heuristicas.TabooSearch.TS(solucionPrueba,solucionPrueba,evaluacion(solucionPrueba),vecindad,evaluacion,44,10,7)
    except Exception as e:
        return(f"Failed to run target heuristic: {e}.  Signature def SA(solution,best_sol, best_score, generate_neighbour(), evaluate_solution(), TEMP, MIN_TEMP, cooling_factor)")
    #Cargar heuristicas, retornar resultados de cada una.
    return resultadoTS


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

def evaluarComponentesILSud(Eval, Nb, Perturb,SampleSol):
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
        resultadoSA = Heuristicas.IteratedLocalSearch.ILS(solucionPrueba,solucionPrueba,evaluacion(solucionPrueba),vecindad,perturb,evaluacion,44,0.1) ##44 iteraciones es lo mismo que tiene el de SA, en base a los valores de 1000, 10 y 0,9
    except Exception as e:
        return(f"Failed to run target heuristic: {e}.  Signature def ILS(solution,best_sol, best_score, generate_neighbour(),perturb_solution(), evaluate_solution(), iterations, aceptance_rate)")
    #Cargar heuristicas, retornar resultados de cada una.
    return resultadoSA

def evaluarComponentesTSud(Eval, Nb, Perturb,SampleSol):
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
        resultadoTS = Heuristicas.TabooSearch.TS(solucionPrueba,solucionPrueba,evaluacion(solucionPrueba),vecindad,evaluacion,44,10,7)
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

def generarDiagnostico(context_message):
        tipo, error, tb  = sys.exc_info()
        ultimoFrame = traceback.extract_tb(tb)[-1]
        return (f"{context_message}\n"
                f"Type: {tipo.__name__}\n"
                f"Message: {error}\n"
                f"Line {ultimoFrame.lineno}\n"
                f"Code{ultimoFrame.line}")