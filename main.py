import argparse
import glob
import re
import sys
import time
from dotenv import load_dotenv
import pandas as pd
import Analisis
import Plotter
import numpy as np
#Estos estan aqui porque el codigo generado por IA tiende a necesitarlos. Como no pueden importar nada ellas, esto evita que exploten los solver
import random
import math 
# List y Union tambien estan aqui por el mismo motivo.
from typing import List, Union,Callable, Any, Dict
from scipy.stats import t, fisher_exact
import os
import Optimizacion.PromptSamplerOP as PromptSamplerOP
import json
from Preparacion import Preparacion
from Instancias import DataLoader
from Generador import generador
from Optimizacion import Optimizacion
from io import StringIO


def main():
    #Inicializar entorno
    semilla = 1234
    iteraciones = 3 
    filaMultiplo = 3
    instancias = DataLoader()
    load_dotenv()
    #Modificacion para pruebas, prepara modo batch por defecto. Mas rapido en caso de que el codigo falle
    if len(sys.argv) == 1:
        sys.argv.extend(['K', 'knap', '-p'])
        #sys.argv.extend(['GC', 'graph', '-plt'])
        #sys.argv.extend(['TS', 'travel', '-plt'])
    DATA_PATH= os.path.join(os.path.dirname(__file__), 'Data')
    #Fin modificacion para pruebas
    parser = argparse.ArgumentParser()
    # Este no tiene flag (-  o bien --). Es posicional. Para referencia futura: --help pone los posicionales primero
    parser.add_argument('tipoProblema', type=str, choices=instancias.getClaves().keys(), help='Tipo de problema a procesar.')
    parser.add_argument('problema_ID', nargs='?',default=None,help="ID del problema a optimizar: Formato: Tipo_dataset_ID, IDs son equivalentes al nombre de carpeta que contiene los datos de la instancia, incompatible con '-b/--batch'")
    parser.add_argument('-b', '--batch', action= 'store_true', dest='batch',help='Realizar operaciones con todos los datos y problemas disponibles de forma automatica')
    parser.add_argument('-p', '--prep',action='store_true', dest='prep', help='Realiza preparacion de problemas')
    parser.add_argument('-o', '--opt',action='store_true', dest='opt', help='Optimizar. Espera preparacion previa - Usar despues the -p o --prep. sin solucion conocida')
    parser.add_argument('-oi', '--opti',action='store_true', dest='opti', help='Optimizar con inspiraciones. Espera preparacion previa - Usar despues the -p o --prep.')
    parser.add_argument('-np', '--noprep',action='store_true', dest='skip_prep', help='Optimizar sin preparar antes, incompatible con --prep/-p y --opt/-o')
    parser.add_argument('-plt', '--plot',action='store_true', dest='plot', help='Procesar resultados')
    parser.add_argument('-s', '--seed',action='store_true', dest='seed', help='Semilla para operaciones aleatorias a utilizar')
     ## puede que lo podamos reciclar para otra cosa, sino eliminar
    parser.add_argument('-qt','--quicktest',action='store_true', dest='quicktest', help='Probar funcionalidad de componentes generados')
    args=parser.parse_args()
    tipoProblema = args.tipoProblema
    if args.seed: semilla = args.seed
    os.makedirs(DATA_PATH, exist_ok=True)
    pipeline = "prep"
    if args.opti:
        pipeline = "full-i"
    if args.opt:
        pipeline = "full"
    if args.skip_prep:
        pipeline = "directa"
            
    problemasPath = os.path.join(DATA_PATH, f'problemas-{tipoProblema}.jsonl')
    dataStructPath = os.path.join(DATA_PATH, f"dataStruct-{tipoProblema}.jsonl")
    inspiracionPath = os.path.join(DATA_PATH, f'inspiraciones-{tipoProblema}.jsonl')
    componentesPath = os.path.join(DATA_PATH, f'componentes-{tipoProblema}-{pipeline}.jsonl')
    feedbackPath = os.path.join(DATA_PATH,f'feedback-{tipoProblema}-{pipeline}.jsonl')
    resultsPath = os.path.join(DATA_PATH,f'resultados-{tipoProblema}-{pipeline}.jsonl')
    
    instancias.cargarProblemas(tipoProblema)
    schemaEstandar, dataclassProblema, dataclassProblemaInst = instancias.getSchema(tipoProblema)
    llms = generador()
    llms.cargarLLMs()
    
    if args.plot:
        pipelines = []
        resultados = []
        registroPipelines = []
        archivos = glob.glob(os.path.join(DATA_PATH, f"resultados-{tipoProblema}-*.jsonl"))
    
        if not archivos:
            print("No hay resultados disponibles. Corra el algoritmo primero")
            return 0
        
        for resultsPath in archivos:
            match = re.search(rf"resultados-{re.escape(tipoProblema)}-(.+)\.jsonl$", os.path.basename(resultsPath))
            if match:
                pipelines.append(match.group(1))
                resultados.append(cargarResultados(resultsPath))

        for pipeline, resultDB in zip(pipelines, resultados):
            tablaOutput = os.path.join(DATA_PATH,f'tablasLatex-{pipeline}.tex')
            dfProcesado, fallos,fallosI,fallosS, resultadosAux, correctitud,correctituI,correctitudS = Analisis.procesarResultados(resultDB, instancias)
            Analisis.actualizarDictPipelines(FallosTot=fallos,registroPipelines=registroPipelines,pipeline=pipeline,dfProcesado=dfProcesado,resultadosAux=resultadosAux)
            Analisis.generarFigurasYTablasLatexLocales(dfProcesado, fallos,fallosI,fallosS, resultadosAux,  correctitud,correctituI,correctitudS,  tablaOutput,f"CE-{tipoProblema}-{pipeline}")            
            pd.set_option('display.float_format', lambda x: '%.0000f' % x) #Poco elegante pero funciona
        Analisis.plotsGlobales(registroPipelines)
        return 0

    if args.quicktest:
        with open(problemasPath, 'r') as f:
            problemaDB = pd.read_json(StringIO(f.read()), lines=True)
        if os.path.exists(componentesPath) and os.path.exists(resultsPath):
            with open(componentesPath, 'r') as f:
                   componenteDB = pd.read_json(StringIO(f.read()), lines=True)
            with open(resultsPath, 'r') as f:
                   resultDB = pd.read_json(StringIO(f.read()), lines=True)
        else:
            componenteDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Evaluacion', 'Vecindad', 'Perturbacion','Version'])
            resultDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Evaluacion', 'Vecindad', 'Perturbacion', 'Resultados','Solucion','Valor Optimo', 'Metaheuristica', 'Tiempo'])
        for problema in problemaDB.itertuples(index=False):
            resultDB = comprobarComponentesUD(problema,componenteDB,resultDB)
        resultDB.to_json(resultsPath,orient='records',lines=True)
        return 0

    if(args.problema_ID and args.batch):
        parser.error("El argumento -b/--batch) no puede ser usado si se define una ID de problema")
    if(args.prep or args.opt) and args.skip_prep:
        parser.error("El argumento -np/--noprep no es compatible con --prep/-p y/o --opt/-o'")
    if args.problema_ID:
        problema_ID = args.problema_ID
        if args.prep:
            print("Datos inicializados. Iniciando preparación")
            prepararIndividual(instancias,llms,problemasPath,dataStructPath,problema_ID,schemaEstandar, dataclassProblema) #NGuarda Datos directamente. Se usa el archivo CSV para pasar datos a traves del sistema
            #probar los dataStruct para saber si funcionan bien por medio de hacerlos cargar cada experimento?. 
        if args.opti:
                componenteDB, feedbackDB, resultDB, InspiracionDB = cargarDBs(componentesPath,resultsPath,feedbackPath, inspiracionPath)
                with open(problemasPath, 'r') as f:
                    problemaDB = pd.read_json(StringIO(f.read()), lines=True)
                with open(dataStructPath, 'r') as f:
                    dataStructDB = pd.read_json(StringIO(f.read()), lines=True)
                filas_serie = problemaDB[problemaDB['Instancia'].str.startswith(problema_ID, na=False)] ## poco eficiente, pero como solo se hace unas pocas veces no importa. .iloc[0] es para que nos entregue la fila como serie
                
                if filas_serie.empty: 
                    print("Id de problema %s no encontrado, preparelo primero o revise los datos de entrada", problema_ID)
                    return 0
                print("Datos inicializados. Iniciando optimizacion")
                for problema in filas_serie.itertuples(index=False):
                    problema_ID = problema.Instancia
                    filas_struct = dataStructDB[dataStructDB['Instancia'] == problema_ID].iloc[0]
                    schemaGenerado = problema.Respuesta
                    convertidorText = filas_struct['FuncionDeCarga'] 
                    try:
                        problemData = Optimizacion.cargarDatosProblema(schemaGenerado,dataclassProblema,convertidorText)
                    except Exception as e: #En caso de que use la instancia directamente, no hay forma de saber que va a intentar
                         problemData = Optimizacion.cargarDatosProblema(schemaGenerado,dataclassProblemaInst,convertidorText)
                    if(not problemData):
                        print(f"Error al inicializar datos del problema {problema_ID}, revise como fue preparado antes de continuar")
                        continue;
                    print(problemData)
                    componenteDB, feedbackDB, resultDB, InspiracionDB = Optimizacion.optimizarProblemaPreparadoConInspiraciones(problema,schemaEstandar,problemData, componenteDB,resultDB,feedbackDB,InspiracionDB, iteraciones, semilla)
                    #crear funcion para probar componentes generados para el problema.
                    componenteDB.to_json(componentesPath,orient='records',lines=True)
                    feedbackDB.to_json(feedbackPath,orient='records',lines=True)
                    resultDB.to_json(resultsPath,orient='records',lines=True)
                    InspiracionDB.to_json(inspiracionPath,orient='records',lines=True)   
                return 0
        if args.opt:
                componenteDB, feedbackDB, resultDB, InspiracionDB = cargarDBs(componentesPath,resultsPath,feedbackPath, inspiracionPath)
                with open(problemasPath, 'r') as f:
                    problemaDB = pd.read_json(StringIO(f.read()), lines=True)
                with open(dataStructPath, 'r') as f:
                    dataStructDB = pd.read_json(StringIO(f.read()), lines=True)
                filas_serie = problemaDB[problemaDB['Instancia'].str.startswith(problema_ID, na=False)] ## poco eficiente, pero como solo se hace unas pocas veces no importa. .iloc[0] es para que nos entregue la fila como serie
                
                if filas_serie.empty: 
                    print("Id de problema %s no encontrado, preparelo primero o revise los datos de entrada", problema_ID)
                    return 0
                print("Datos inicializados. Iniciando optimizacion")
                for problema in filas_serie.itertuples(index=False):
                    problema_ID = problema.Instancia
                    filas_struct = dataStructDB[dataStructDB['Instancia'] == problema_ID].iloc[0]
                    schemaGenerado = problema.Respuesta
                    convertidorText = filas_struct['FuncionDeCarga'] 
                    try:
                        problemData = Optimizacion.cargarDatosProblema(schemaGenerado,dataclassProblema,convertidorText)
                    except Exception as e: #En caso de que use la instancia directamente, no hay forma de saber que va a intentar
                         problemData = Optimizacion.cargarDatosProblema(schemaGenerado,dataclassProblemaInst,convertidorText)
                    if(not problemData):
                        print(f"Error al inicializar datos del problema {problema_ID}, revise como fue preparado antes de continuar")
                        continue;
                    print(problemData)
                    componenteDB, feedbackDB, resultDB, InspiracionDB = Optimizacion.optimizarProblemaPreparado(problema,schemaEstandar,problemData, componenteDB,resultDB,feedbackDB,InspiracionDB, iteraciones, semilla)
                    #crear funcion para probar componentes generados para el problema.
                    componenteDB.to_json(componentesPath,orient='records',lines=True)
                    feedbackDB.to_json(feedbackPath,orient='records',lines=True)
                    resultDB.to_json(resultsPath,orient='records',lines=True)
                    InspiracionDB.to_json(inspiracionPath,orient='records',lines=True)         
        if args.skip_prep and not args.opt and not args.prep:
            instancias = instancias.getInstancias(problema_ID)
            if len(instancias) == 0 : 
                print("Id de problema %s no encontrado, definalo primero o revise los datos de entrada", problema_ID)
                return 0
            componenteDBNP, feedbackDBNP, resultDBNP = cargarDBs(componentesPath,resultsPath,feedbackPath)
            print("Datos inicializados. Iniciando Iniciando optimizacion")
            for instancia in instancias:
                componenteDBNP, feedbackDBNP, resultDBNP = Optimizacion.optimizarProblemaSinPreparar(instancia.claveInstancia,instancia.problem, instancia.parsedSolution,instancia.objectiveScore, componenteDBNP,resultDBNP,feedbackDBNP, iteraciones) 
                componenteDBNP.to_json(componentesPath,orient='records',lines=True)
                feedbackDBNP.to_json(feedbackPath,orient='records',lines=True)
                resultDBNP.to_json(resultsPath,orient='records',lines=True)
            return 0

    if args.batch and not args.problema_ID:
        if args.prep:
            print("Datos inicializados. Iniciando preparación")
            if tipoProblema == "GC":
                Preparacion.prepararBatch(instancias,llms, problemasPath, tipo="graph_coloring")
            elif tipoProblema == "K":
                Preparacion.prepararBatch(instancias,llms, problemasPath, tipo="knapsack")
        if args.opti:
            print("Datos inicializados. Iniciando optimización")
            with open(problemasPath, 'r') as f:
                problemaDB = pd.read_json(StringIO(f.read()), lines=True)
            componenteDB, feedbackDB, resultDB = cargarDBs(componentesPath,resultsPath,feedbackPath)
            problemasFiltrados = problemaDB.iloc[filaMultiplo::filaMultiplo]
            for problema in problemasFiltrados.itertuples(index=False):    
                componenteDB, feedbackDB, resultDB = Optimizacion.optimizarProblemaPreparadoConInspiraciones(problema, componenteDB,resultDB,feedbackDB, iteraciones)
                componenteDB.to_json(componentesPath,orient='records',lines=True)
                feedbackDB.to_json(feedbackPath,orient='records',lines=True)
                resultDB.to_json(resultsPath,orient='records',lines=True)
        if args.opt:
            print("Datos inicializados. Iniciando optimización")
            with open(problemasPath, 'r') as f:
                problemaDB = pd.read_json(StringIO(f.read()), lines=True)
            componenteDB, feedbackDB, resultDB = cargarDBs(componentesPath,resultsPath,feedbackPath)
            problemasFiltrados = problemaDB.iloc[filaMultiplo::filaMultiplo]
            for problema in problemasFiltrados.itertuples(index=False):    
                componenteDB, feedbackDB, resultDB = Optimizacion.optimizarProblemaPreparado(problema, componenteDB,resultDB,feedbackDB, iteraciones)
                componenteDB.to_json(componentesPath,orient='records',lines=True)
                feedbackDB.to_json(feedbackPath,orient='records',lines=True)
                resultDB.to_json(resultsPath,orient='records',lines=True)

        if args.skip_prep and not args.opt and not args.prep:
            componenteDBNP, feedbackDBNP, resultDBNP = cargarDBs(componentesPath,resultsPath,feedbackPath)
            instancias = instancias.getAllInstancias()
            if len(instancias) == 0: 
                print("No se encontraron instancias de problemas para optimizar, revise que existen datos en %s con formato Tipo_dataset_ID, donde Tipo, dataset e ID son carpetas anidades en ese orden y/o que existen problemas definidos en un archivo que termine en *sample.csv dentro de %s", DATA_PATH)
                return 0
            print("Datos inicializados. Iniciando Iniciando optimizacion")
            for instancia in instancias:
                componenteDBNP, feedbackDBNP, resultDBNP = Optimizacion.optimizarProblemaSinPreparar(instancia.claveInstancia,instancia.problem, instancia.parsedSolution,instancia.objectiveScore, componenteDBNP,resultDBNP,feedbackDBNP, iteraciones) 
                componenteDBNP.to_json(componentesPath,orient='records',lines=True)
                feedbackDBNP.to_json(feedbackPath,orient='records',lines=True)
                resultDBNP.to_json(resultsPath,orient='records',lines=True)
            return 0
       

    with open(problemasPath, 'r') as f:
        problemaDB = pd.read_json(StringIO(f.read()), lines=True)

    return 0

def prepararIndividual(instancias,llms,problemasPath, dataStructPath,instancia, schema, dataclassProblema):
    respuesta, feedback, convertidor = Preparacion.extraerIndividual(instancias,llms,problemasPath, dataStructPath, instancia,schema, dataclassProblema)
    print("RESPUESTA final: \n" + respuesta + "---------------------- \n FEEDBACK mas reciente: \n" + feedback)
    return respuesta,feedback, convertidor

def cargarDBs(componentesPath,resultPath,feedbackPath, inspiracionPath):
    if os.path.exists(componentesPath): 
        with open(componentesPath, 'r') as f:
           componenteDB = pd.read_json(StringIO(f.read()), lines=True)
    else: componenteDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Evaluacion', 'Vecindad', 'Perturbacion','SolucionPrueba','Version'])
    if os.path.exists(feedbackPath): 
        with open(feedbackPath, 'r') as f:
            feedbackDB = pd.read_json(StringIO(f.read()), lines=True)
    else: feedbackDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Componente','Version', 'Feedback'])
    if os.path.exists(resultPath): 
        with open(resultPath, 'r') as f:
            resultDB = pd.read_json(StringIO(f.read()), lines=True)
    else:resultDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Evaluacion', 'Vecindad', 'Perturbacion', 'Resultados','Solucion','Valor Optimo', 'Metaheuristica', 'Tiempo'])
    if os.path.exists(inspiracionPath): 
        with open(inspiracionPath, 'r') as f:
            inspiracionDB = pd.read_json(StringIO(f.read()), lines=True)
    else: inspiracionDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Evaluacion', 'Vecindad', 'Perturbacion','SolucionPrueba','Version'])
    return componenteDB,feedbackDB,resultDB, inspiracionDB

def comprobarComponentesCD(problema,componentes:pd.DataFrame,resultDB: pd.DataFrame):
            problemaID, defProblema, _, _, seedPrompt = PromptSamplerOP.generateSeedPrompt(problema)
            resultadoSA, tiempoSA = cronometrarFuncion(Optimizacion.evaluarComponentesSA ,componentes)
            resultadoILS, tiempoILS = cronometrarFuncion(Optimizacion.evaluarComponentesILS, componentes)
            resultadoTS, tiempoTS = cronometrarFuncion(Optimizacion.evaluarComponentesTS, componentes)
            resultDB = guardarResultado(problemaID,resultDB, componentes['Representacion'],componentes['Evaluacion'],componentes['Vecindad'],componentes['Perturbacion'],resultadoSA,"NA", "NA", "SA", tiempoSA)
            resultDB = guardarResultado(problemaID,resultDB, componentes['Representacion'],componentes['Evaluacion'],componentes['Vecindad'],componentes['Perturbacion'],resultadoILS,"NA", "NA", "ILS", tiempoILS)
            resultDB = guardarResultado(problemaID,resultDB, componentes['Representacion'],componentes['Evaluacion'],componentes['Vecindad'],componentes['Perturbacion'],resultadoTS,"NA", "NA", "TS", tiempoTS)
            return resultDB

def comprobarComponentesUD(problema,componentes:pd.DataFrame,resultDB: pd.DataFrame):
    problemaID, _, _, _, _ = PromptSamplerOP.generateSeedPrompt(problema)
    resultadoSA, tiempoSA = cronometrarFuncion(Optimizacion.evaluarComponentesSA, componentes)
    resultadoILS, tiempoILS = cronometrarFuncion(Optimizacion.evaluarComponentesILS, componentes)
    resultadoTS, tiempoTS = cronometrarFuncion(Optimizacion.evaluarComponentesTS, componentes)
    print(f"Resultado SA: {resultadoSA}")
    print(f"Resultado ILS: {resultadoILS}")
    print(f"Resultado TS: {resultadoTS}")
    resultDB = guardarResultado(problemaID,resultDB, componentes['Representacion'],componentes['Evaluacion'],componentes['Vecindad'],componentes['Perturbacion'],resultadoSA,"NA", "NA", "SA", tiempoSA)
    resultDB = guardarResultado(problemaID,resultDB, componentes['Representacion'],componentes['Evaluacion'],componentes['Vecindad'],componentes['Perturbacion'],resultadoILS,"NA", "NA", "ILS", tiempoILS)
    resultDB = guardarResultado(problemaID,resultDB, componentes['Representacion'],componentes['Evaluacion'],componentes['Vecindad'],componentes['Perturbacion'],resultadoTS,"NA", "NA", "TS", tiempoTS)      
    return resultDB



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

def auxParsearInf(val):
    if val == 'Infinity':
        return np.inf
    elif val == '-Infinity':
        return -np.inf
    elif val == 'NaN':
        return np.nan
    raise ValueError(f"Valor desconcido: {val}")

def cargarResultados(path):
    datos = []
    with open(path, 'r') as f:
        for line_number, line in enumerate(f, 1):
            try:
                record = json.loads(
                    line.strip(), 
                    parse_constant=auxParsearInf
                )
                datos.append(record)
            except Exception as e:
                print(f"Fallo en {line_number}: {e} -> Datos: '{line.strip()}'")
    return pd.DataFrame(datos)


def compararResultados(metricasRendimiento1, metricasRendimiento2,desempeñoPorSolver1, desempeñoPorSolver2,output):
            comparacion = metricasRendimiento1.merge(
                metricasRendimiento2,
                on='Metaheuristica',
                suffixes=('_FP', '_Control')
            )
            # Calcular delta (error absoluto negativo implica que FP es mas preciso == mejor)
            comparacion['Delta_Error_Abs']  = comparacion['Promedio_Error_Abs_FP']  - comparacion['Promedio_Error_Abs_Control']
            comparacion['Delta_Error_Pct']  = comparacion['Promedio_Error_Porcentual_FP'] - comparacion['Promedio_Error_Porcentual_Control']
            # Calculate Intervalo de Confianza de delta
            alpha = 0.05
            df = (
                comparacion['exitosTotales_FP'] +
                comparacion['exitosTotales_Control'] - 2
            )   
            tCrit = t.ppf(1 - alpha/2, df)
            comparacion['IC_Delta_Error_Abs_Menor'] = (
                comparacion['Delta_Error_Abs'] -
                tCrit * np.sqrt(
                    (comparacion['Desviacion_Estandar_Abs_FP']**2 / comparacion['exitosTotales_FP']) +
                    (comparacion['Desviacion_Estandar_Abs_Control']**2 / comparacion['exitosTotales_Control'])
                )
            )

            comparacion['IC_Delta_Error_Abs_Maximo'] = (
                comparacion['Delta_Error_Abs'] +
                tCrit * np.sqrt(
                    (comparacion['Desviacion_Estandar_Abs_FP']**2 / comparacion['exitosTotales_FP']) +
                    (comparacion['Desviacion_Estandar_Abs_Control']**2 / comparacion['exitosTotales_Control'])
                )
            )

            comparacion['IC_Delta_Error_Pct_Menor'] = (
                comparacion['Delta_Error_Pct'] -
                tCrit * np.sqrt(
                    (comparacion['Desviacion_Estandar_Porcentual_FP']**2 / comparacion['exitosTotales_FP']) +
                    (comparacion['Desviacion_Estandar_Porcentual_Control']**2 / comparacion['exitosTotales_Control'])
                )
            )
            comparacion['IC_Delta_Error_Pct_Maximo'] = (
                comparacion['Delta_Error_Pct'] +
                tCrit * np.sqrt(
                    (comparacion['Desviacion_Estandar_Porcentual_FP']**2 / comparacion['exitosTotales_FP']) +
                    (comparacion['Desviacion_Estandar_Porcentual_Control']**2 / comparacion['exitosTotales_Control'])
                )
            )
            print('--- Comparacion entre pipas de proceso, Con Definicion y Sin Definicion ---')
            print(comparacion[['Metaheuristica','Delta_Error_Pct', 'IC_Delta_Error_Pct_Menor', 'IC_Delta_Error_Pct_Maximo','Delta_Error_Abs', 'IC_Delta_Error_Abs_Menor', 'IC_Delta_Error_Abs_Maximo']])
            latexComp = comparacion[['Metaheuristica','Delta_Error_Pct', 'IC_Delta_Error_Pct_Menor', 'IC_Delta_Error_Pct_Maximo','Delta_Error_Abs', 'IC_Delta_Error_Abs_Menor', 'IC_Delta_Error_Abs_Maximo']].to_latex(index=False, column_format='lrrrrrrr', caption='Comparación entre sistema con definicion (CD) y sin definicion (SD).', label='tab:comparacion_pipelines')
            fisher_df = desempeñoPorSolver1.merge(desempeñoPorSolver2,on='Metaheuristica',suffixes=('_FP', '_Control'))
            p_bilateral = []
            p_mejor = []
            for _, row in fisher_df.iterrows():
                table = [
                    [row['TotalExitos_FP'],   row['TotalFallos_FP']],
                    [row['TotalExitos_Control'], row['TotalFallos_Control']]
                ]

                # Fisher bilateral (H0: No hay diferencia). Tiene que ser menor a 0.05 para ser cierto
                p2 = fisher_exact(table, alternative='two-sided').pvalue
                p_bilateral.append(p2)

                # Fisher hacia un solo lado (H1: FP funciona mejor que Control)
                p1 = fisher_exact(table, alternative='greater').pvalue
                p_mejor.append(p1)

            fisher_df['PValue_Bilateral'] = p_bilateral
            fisher_df['PValue_FPMejor'] = p_mejor
            print(fisher_df[['Metaheuristica',
                 'TotalExitos_FP', 'TotalFallos_FP',
                 'TotalExitos_Control', 'TotalFallos_Control',
                 'PValue_Bilateral', 'PValue_FPMejor']])
            latexFish = fisher_df[['Metaheuristica',
                 'TotalExitos_FP', 'TotalFallos_FP',
                 'TotalExitos_Control', 'TotalFallos_Control',
                 'PValue_Bilateral', 'PValue_FPMejor']].to_latex(index=False, column_format='lrrrrrrr', caption='Comparación entre sistemas con Fisher.', label='tab:prueba_hipotesis' )
            Plotter.graficos_globales(comparacion, fisher_df)
            with open(output, 'a', encoding='utf-8') as f:
                f.write(latexComp)
                f.write('----------------------\n\n')
                f.write(latexFish)

def procesarResultadosPorIteracion(resultados: pd.DataFrame,dfProcesado: pd.DataFrame, output, pipeline, instancias: DataLoader, iteraciones):
    filasPorIteracion = 4
    for iteracion in range(iteraciones):
        subset = resultados.groupby('ID_Problema').apply(
            lambda g: g.iloc[iteracion * filasPorIteracion : (iteracion + 1) * filasPorIteracion]
        ).reset_index(drop=True)

        if subset.empty:
            print(f"[iteracion {iteracion + 1}]: Sin datos, ignorando.")
            continue
        subset = subset.groupby('ID_Problema').filter(lambda g: len(g) == filasPorIteracion)

        if subset.empty:
            print(f"[iteracion {iteracion +1}]: Todos los grupos incompletos, ignorando.")
            continue

        print(f"\n{'='*40}")
        print(f"  Resultados iteracion {iteracion +1}")
        print(f"{'='*40}")
        Analisis.procesarResultados(subset, output, f"{pipeline}_iteracion_{iteracion +1}", instancias)



def cronometrarFuncion(func: Callable, *args, **kwargs) -> tuple[float, any]:
    inicio = time.perf_counter()
    resultado = func(*args, **kwargs)
    fin = time.perf_counter()
    tiempo = fin - inicio
    return resultado, tiempo

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


main()