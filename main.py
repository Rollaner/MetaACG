import ast
import argparse
import numbers
import sys
import time
from dotenv import load_dotenv
import pandas as pd
import Plotter
import numpy as np
#Estos estan aqui porque el codigo generado por IA tiende a necesitarlos. Como no pueden importar nada ellas, esto evita que exploten los solver
import random
import math 
# List y Union tambien estan aqui por el mismo motivo.
from scipy.stats import t, fisher_exact
import os
import Optimizacion.PromptSamplerOP as PromptSamplerOP
import json
from typing import List, Union,Callable, Any, Dict
from Preparacion import Preparacion
from Instancias import DataLoader
from Generador import generador
from Optimizacion import Optimizacion


def main():
    semilla = 1234
    iteraciones = 3 #Genracion se atasca muy seguido. Se prueba sin reiniciar, luego probabos con reinicio. 
    filaInicio = 3
    filaMultiplo = 3
    tipoProblema = "K"
    #Inicializar datos
    load_dotenv()
    #Modificacion para pruebas, prepara modo batch por defecto. Estas lineas se tienen que eliminar cuando se empieze a optimizar
    if len(sys.argv) == 1:
        sys.argv.extend(['knapsack_hard_dataset_in_house_24_11', '-o'])
    pathDB= os.path.join(os.path.dirname(__file__), 'Data')
    #Fin modificacion para pruebas
    os.makedirs(pathDB, exist_ok=True)
    problemasPath = os.path.join(pathDB, f'problemas-{tipoProblema}.jsonl')
    componentesPath = os.path.join(pathDB, f'componentes-SR-{tipoProblema}.jsonl')
    feedbackPath = os.path.join(pathDB,f'feedback-SR-{tipoProblema}.jsonl')
    resultPath = os.path.join(pathDB,f'resultados-SR-{tipoProblema}.jsonl')
    componentesPathNP = os.path.join(pathDB, f'componentes-SR-{tipoProblema}-NP.jsonl')
    feedbackPathNP = os.path.join(pathDB,f'feedback-SR-{tipoProblema}-NP.jsonl')
    resultPathNP = os.path.join(pathDB,f'resultados-SR-{tipoProblema}-NP.jsonl')
    instancias = DataLoader()
    instancias.cargarProblemas()
    llms = generador()
    llms.cargarLLMs()
    parser = argparse.ArgumentParser()
    # Este no tiene flag (-  o bien --). Es posicional. Para referencia futura: --help pone los posicionales primero
    parser.add_argument('problema_ID', nargs='?',default=None,help="ID del problema a optimizar: Formato: Tipo_dataset_ID, IDs son equivalentes al nombre de carpeta que contiene los datos de la instancia, incompatible con '-b/--batch'")
    parser.add_argument('-t', '--type', type=str, choices=['K','GC'], help='Tipo de problema para procesamiento en Batch. (K)napsack, (GC) Graph Coloring')
    parser.add_argument('-b', '--batch', action= 'store_true', dest='batch',help='Realizar operaciones con todos los datos y problemas disponibles de forma automatica')
    parser.add_argument('-p', '--prep',action='store_true', dest='prep', help='Realiza preparacion de problemas en batch, no optimiza')
    parser.add_argument('-o', '--opt',action='store_true', dest='opt', help='Solo optimizar, pero espera preparacion previa- Usar despues the -p o --prep. sin solucion conocida')
    parser.add_argument('-np', '--noprep',action='store_true', dest='skip_prep', help='Optimizar pero sin preparar antes, incompatible con --prep/-p y --opt/-o')
    parser.add_argument('-plt', '--plot',action='store_true', dest='plot', help='Procesar resultados')
    
    ## puede que lo podamos reciclar para otra cosa, sino eliminar
    parser.add_argument('-qt','--quicktest',action='store_true', dest='quicktest', help='Probar funcionalidad de componentes generados')
    args=parser.parse_args()

    if args.plot:
        problemaDB = pd.read_json(problemasPath,lines=True)
        if os.path.exists(componentesPath) and os.path.exists(resultPath) and os.path.exists(feedbackPath):
            resultDB = cargarResultados(os.path.join(pathDB,'resultados-SR-K.jsonl'))
            resultControlDB = cargarResultados(os.path.join(pathDB,'resultados-SR-K-SD.jsonl'))
            resultUDDB = cargarResultados(os.path.join(pathDB,'resultados-SR-K-PR.jsonl'))
            resultGCDB = cargarResultados(os.path.join(pathDB,'resultados-SR-GC.jsonl'))
            resultGCDBH = cargarResultados(os.path.join(pathDB,'resultados-SR-GC-H.jsonl'))
            resultKDBH = cargarResultados(os.path.join(pathDB,'resultados-SR-K-H.jsonl'))
            resultGCDBHUD = cargarResultados(os.path.join(pathDB,'resultados-SR-GC-HUD.jsonl'))
            output = os.path.join(pathDB,'tablasLatex.tex')
            #fallosCD, dfProcesadoCD, desempeñoPorSolverCD, metricasRendimientoCD = procesarResultados(resultKDB, output,"CDK","K")
            #fallosKH, dfProcesadoKH, desempeñoPorSolverKH, metricasRendimientoKH = procesarResultados(resultKDBH, output,"CDK-H","KH", problemaDB)
            #fallosC, dfProcesadoC, desempeñoPorSolverC, metricasRendimientoC = procesarResultados(resultControlDB, output,"SDK","K")
            #fallosUD, dfProcesadoUD, desempeñoPorSolverUD, metricasRendimientoUD = procesarResultados(resultUDDB, output,"UDK","K")
            #fallosGC, dfProcesadoGC, desempeñoPorSolverGC, metricasRendimientoGC = procesarResultados(resultGCDB, output,"CDGC","GC")

            fallosGCH, dfProcesadoGCH, desempeñoPorSolverGCH, metricasRendimientoGCH = procesarResultados(resultGCDBH, output,"CDGC-H-L","GCH", problemaDB)
            pd.set_option('display.float_format', lambda x: '%.0000f' % x) #Poco elegante pero funciona
            #compararResultados(metricasRendimientoCD,metricasRendimientoC,desempeñoPorSolverCD, desempeñoPorSolverC,output)
            #Necesita ser cambiado. De momento la comparacion FP y Control esta hardcodeada
            #compararResultados(metricasRendimientoKH,metricasRendimientoGCH,desempeñoPorSolverKH, desempeñoPorSolverGCH,output)
        else:
            print("No hay resultados disponibles. Corra el algoritmo primero")
        return 0

    if args.quicktest:
        problemaDB = pd.read_json(problemasPath,lines=True)
        if os.path.exists(componentesPath) and os.path.exists(resultPath):
            componenteDB = pd.read_json(componentesPath,lines=True)
            resultDB = pd.read_json(resultPath,lines=True)
        else:
            componenteDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Evaluacion', 'Vecindad', 'Perturbacion','Version'])
            resultDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Evaluacion', 'Vecindad', 'Perturbacion', 'Resultados','Solucion','Valor Optimo', 'Metaheuristica', 'Tiempo'])
        for problema in problemaDB.itertuples(index=False):
            resultDB = comprobarComponentesUD(problema,componenteDB,resultDB)
        resultDB.to_json(resultPath,orient='records',lines=True)
        return 0

    if(args.problema_ID and args.batch):
        parser.error("El argumento -b/--batch) no puede ser usado si se define una ID de problema")
    if(args.prep or args.opt) and args.skip_prep:
        parser.error("El argumento -np/--noprep no es compatible con --prep/-p y/o --opt/-o'")
    if args.problema_ID:
        problema_ID = args.problema_ID
        print(problema_ID)
        if args.prep:
            print("Datos inicializados. Iniciando preparación")
            listaDatos = prepararIndividual(instancias,llms,problemasPath,problema_ID) #Nos interesan 1, 5,6,7,8 y 9. ID, Respuesta, Feedback, contenidos, solucion, valor solucion respectivamente
        if args.opt:
                componenteDB, feedbackDB, resultDB = cargarDBs(componentesPath,resultPath,feedbackPath)
            #if listaDatos:
            #    print("Datos inicializados. Iniciando optimizacion")
            #    for datos in listaDatos:
            #        problema = datos[5] 
            #        solucion = datos[8]
            #        componenteDB, feedbackDB, resultDB = Optimizacion.optimizarProblemaPreparado(problema_ID, problema, solucion, componenteDB,resultDB,feedbackDB, iteraciones)
            #        componenteDB.to_json(componentesPath,orient='records',lines=True)
            #        feedbackDB.to_json(feedbackPath,orient='records',lines=True)
            #        resultDB.to_json(resultPath,orient='records',lines=True)
            #    return 0
            #else:
                problemaDB = pd.read_json(problemasPath,lines=True)
                filas_serie = problemaDB[problemaDB['Instancia'].str.startswith(problema_ID, na=False)] ## poco eficiente, pero como solo se hace unas pocas veces no importa. .iloc[0] es para que nos entregue la fila como serie
                if filas_serie.empty: 
                    print("Id de problema %s no encontrado, preparelo primero o revise los datos de entrada", problema_ID)
                    return 0
                print("Datos inicializados. Iniciando optimizacion")
                for problema in filas_serie.itertuples(index=False):
                    componenteDB, feedbackDB, resultDB = Optimizacion.optimizarProblemaPreparadoDB(problema, componenteDB,resultDB,feedbackDB, iteraciones)
                    componenteDB.to_json(componentesPath,orient='records',lines=True)
                    feedbackDB.to_json(feedbackPath,orient='records',lines=True)
                    resultDB.to_json(resultPath,orient='records',lines=True)
                return 0         
        if args.skip_prep and not args.opt and not args.prep:
            instancias = instancias.getInstancias(problema_ID)
            if len(instancias) == 0 : 
                print("Id de problema %s no encontrado, definalo primero o revise los datos de entrada", problema_ID)
                return 0
            componenteDBNP, feedbackDBNP, resultDBNP = cargarDBs(componentesPathNP,resultPathNP,feedbackPathNP)
            print("Datos inicializados. Iniciando Iniciando optimizacion")
            for instancia in instancias:
                componenteDBNP, feedbackDBNP, resultDBNP = Optimizacion.optimizarProblemaSinPreparar(instancia.claveInstancia,instancia.problem, instancia.parsedSolution, componenteDBNP,resultDBNP,feedbackDBNP, iteraciones) # Revisar logica para que efectivamente trabaje con los problemas en bruto. Parece que de momento utiliza los mismos que el sistema convencional
                componenteDBNP.to_json(componentesPath,orient='records',lines=True)
                feedbackDBNP.to_json(feedbackPath,orient='records',lines=True)
                resultDBNP.to_json(resultPath,orient='records',lines=True)
            return 0

    if args.batch and not args.problema_ID:
        if args.prep:
            print("Datos inicializados. Iniciando preparación")
            if tipoProblema == "GC":
                Preparacion.prepararBatch(instancias,llms, problemasPath, tipo="graph_coloring")
            elif tipoProblema == "K":
                Preparacion.prepararBatch(instancias,llms, problemasPath, tipo="knapsack")
        if args.opt:
            print("Datos inicializados. Iniciando optimización")
            problemaDB = pd.read_json(problemasPath,lines=True)
            componenteDB, feedbackDB, resultDB = cargarDBs(componentesPath,resultPath,feedbackPath)
            problemasFiltrados = problemaDB.iloc[filaMultiplo::filaMultiplo]
            for problema in problemasFiltrados.itertuples(index=False):    
                componenteDB, feedbackDB, resultDB = Optimizacion.optimizarProblemaPreparado(problema, componenteDB,resultDB,feedbackDB, iteraciones)
                componenteDB.to_json(componentesPath,orient='records',lines=True)
                feedbackDB.to_json(feedbackPath,orient='records',lines=True)
                resultDB.to_json(resultPath,orient='records',lines=True)

        if args.skip_prep and not args.opt and not args.prep:
            componenteDBNP, feedbackDBNP, resultDBNP = cargarDBs(componentesPathNP,resultPathNP,feedbackPathNP)
            instancias = instancias.getAllInstancias()
            if len(instancias) == 0: 
                print("No se encontraron instancias de problemas para optimizar, revise que existen datos en %s con formato Tipo_dataset_ID, donde Tipo, dataset e ID son carpetas anidades en ese orden y/o que existen problemas definidos en un archivo que termine en *sample.csv dentro de %s", pathDB)
                return 0
            print("Datos inicializados. Iniciando Iniciando optimizacion")
            for instancia in instancias:
                componenteDBNP, feedbackDBNP, resultDBNP = Optimizacion.optimizarProblemaSinPreparar(instancia.claveInstancia,instancia.problem, instancia.parsedSolution, componenteDBNP,resultDBNP,feedbackDBNP, iteraciones) # Revisar logica para que efectivamente trabaje con los problemas en bruto. Parece que de momento utiliza los mismos que el sistema convencional
                componenteDBNP.to_json(componentesPath,orient='records',lines=True)
                feedbackDBNP.to_json(feedbackPath,orient='records',lines=True)
                resultDBNP.to_json(resultPath,orient='records',lines=True)
            return 0
       

    #componenteDB = pd.read_json(componentesPath,lines=True)
    problemaDB = pd.read_json(problemasPath,lines=True)
    #feedbackDB = pd.read_json(feedbackPath,lines=True)
    #resultDB = pd.read_json(resultPath,lines=True)
    #componenteDB.to_json(componentesPath,lines=True)
    #feedbackDB.to_json(feedbackPath,lines=True)
    #resultDB.to_json(resultPath,lines=True)
    return 0

def prepararIndividual(instancias,llms,problemasPath,instancia):
    listaDatos = Preparacion.aplanarIndividual(instancias,llms,problemasPath,instancia) #Nos interesan 5,6,7,8 y 9. Respuesta, Feedback, contenidos, solucion, valor solucion respectivamente
    for datos in listaDatos:
        respuesta,feedback = datos[5], datos[6]
        print("RESPUESTA final: \n" + respuesta + "---------------------- \n FEEDBACK mas reciente: \n" + feedback)
    return listaDatos

def cargarDBs(componentesPath,resultPath,feedbackPath):
    if os.path.exists(componentesPath) and os.path.exists(resultPath) and os.path.exists(feedbackPath):
        componenteDB = pd.read_json(componentesPath,lines=True)
        feedbackDB = pd.read_json(feedbackPath,lines=True)
        resultDB = pd.read_json(resultPath,lines=True)
    else:
        componenteDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Evaluacion', 'Vecindad', 'Perturbacion','SolucionPrueba','Version'])
        feedbackDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Componente','Version', 'Feedback'])
        resultDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Evaluacion', 'Vecindad', 'Perturbacion', 'Resultados','Solucion','Valor Optimo', 'Metaheuristica', 'Tiempo'])
    return componenteDB,feedbackDB,resultDB

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

## Funciones compuestas. Presentes tambien en generador. Hace falta refactorizar

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

def auxParsearInf(val):
    if val == 'Infinity':
        return np.inf
    elif val == '-Infinity':
        return -np.inf
    elif val == 'NaN':
        return np.nan
    raise ValueError(f"Valor desconcido: {val}")

def compararResultados(metricasRendimiento1, metricasRendimiento2,desempeñoPorSolver1, desempeñoPorSolver2,output):
            comparacion = metricasRendimiento1.merge(
                metricasRendimiento2,
                on='Metaheuristica',
                suffixes=('_FP', '_Control')
            )
            # Calcular delta (erros absoluto negativo implica que FP es mas preciso == mejor)
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

def procesarResultados(resultados:pd.DataFrame,output, pipeline, problem, problemas:pd.DataFrame = None):
    pd.set_option('display.float_format', lambda x: '%.0f' % x)
    if problem == "GCH" or problem == "KH":
        pd.set_option('display.max_rows', None)
        dfaux = problemas.drop_duplicates(subset=['Instancia'], keep='first')
        map_optimos = dfaux.set_index('Instancia')['Valor Objetivo']
        map_resultados = dfaux.set_index('Instancia')['Resultado esperado']
        print(dfaux)
        resultados['Valor Optimo'] = resultados['ID_Problema'].map(map_optimos)
        resultados['Solucion'] = resultados['ID_Problema'].map(map_resultados)
    print(resultados['Valor Optimo'])
    resultados['Resultados'] = resultados['Resultados'].apply(normalizar_resultado) #Convierte el resultado en algo parseable
    FallosTot = {
    'Failure_to_evaluate': 0,
    'Failure_to_run_target_heuristic': 0,
    'Failure_to_load': 0,
    'Otros': 0,
    'Total Fallos': 0,
    'Total Exitos': 0
    }
    resultadosFiltrados = resultados[~resultados['ID_Problema'].str.contains('inverted', case=False, na=False)].copy() #Para compensar por error en parser de soluciones. Se eliminan los problemas invertidos
    resultadosFiltrados['Puntaje Real'] = resultadosFiltrados['Resultados'].apply(lambda r: parsearResultado(r, FallosTot))
    dfProcesado = resultadosFiltrados[resultadosFiltrados['Puntaje Real'].notna()].copy()
    print(dfProcesado[['Metaheuristica', 'Resultados', 'Puntaje Real', 'Valor Optimo']])
    latex1 = dfProcesado[['Metaheuristica', 'Resultados', 'Puntaje Real', 'Valor Optimo']].to_latex(index=False, column_format='lrrrr', caption=f'Resultados del procesado {pipeline}.', label=f'tab:{pipeline}_resultados_generales' )
    for key, count in FallosTot.items():
        if key not in ['Total Fallos', 'Total Exitos']:
            print(f"* {key}: {count}")
    totalExperimentos =  len(resultadosFiltrados)
    tasaDeFallos = FallosTot['Total Fallos'] / totalExperimentos
    print(f"Tasa de Fallos {tasaDeFallos:.2%}")
    dfProcesado['Diferencia Absoluta'] = (dfProcesado['Puntaje Real']-dfProcesado['Valor Optimo']).abs()
    if problem == "K" or problem == "KH":
        dfProcesado['Diferencia Absoluta'] = (dfProcesado['Puntaje Real']+dfProcesado['Valor Optimo']).abs() #En base a revision manual, en el caso de los problemas de Knapsack el Puntaje real esta siempre cambiado de signo, pero colapsa a la respuesta optima igual. Se cambio el signo del puntaje real a negativo para tomar en cuenta esto
    dfProcesado['Diferencia Porcentual'] = (dfProcesado['Diferencia Absoluta']/dfProcesado['Valor Optimo']).abs()*100 #Valor optimo de EHOP nunca es cero
    resultadosFiltrados['Mascara Exito'] = resultadosFiltrados['Puntaje Real'].notna()
    metricasRendimiento = dfProcesado.groupby('Metaheuristica').agg(
        exitosTotales = ('Puntaje Real', 'count'),
        Promedio_Error_Abs =('Diferencia Absoluta', 'mean'),
        Desviacion_Estandar_Abs = ('Diferencia Absoluta', 'std'),
        Promedio_Error_Porcentual =('Diferencia Porcentual', 'mean'),
        Desviacion_Estandar_Porcentual = ('Diferencia Porcentual', 'std')
    ).reset_index()
    alpha = 0.05
    tCritico = t.ppf(1-alpha/2, metricasRendimiento['exitosTotales']-1)
    #Para saber que tan generlizables son los resultados, considerando la pequeña muestra
    metricasRendimiento['Intervalo_de_Confianza_Error_Absoluto_Minimo'] = ( metricasRendimiento['Promedio_Error_Abs'] - tCritico * metricasRendimiento['Desviacion_Estandar_Abs']/np.sqrt(metricasRendimiento['exitosTotales']))
    metricasRendimiento['Intervalo_de_Confianza_Error_Absoluto_Maximo'] = ( metricasRendimiento['Promedio_Error_Abs'] + tCritico * metricasRendimiento['Desviacion_Estandar_Abs']/np.sqrt(metricasRendimiento['exitosTotales']))
    metricasRendimiento['Intervalo_de_Confianza_Error_Porcentual_Minimo'] = ( metricasRendimiento['Promedio_Error_Porcentual'] - tCritico * metricasRendimiento['Desviacion_Estandar_Porcentual']/np.sqrt(metricasRendimiento['exitosTotales']))
    metricasRendimiento['Intervalo_de_Confianza_Error_Porcentual_Maximo'] = ( metricasRendimiento['Promedio_Error_Porcentual'] + tCritico * metricasRendimiento['Desviacion_Estandar_Porcentual']/np.sqrt(metricasRendimiento['exitosTotales']))
    print("--- Metricas estandar de desempeño ---")
    print(metricasRendimiento[['Metaheuristica','Promedio_Error_Porcentual', 'Desviacion_Estandar_Porcentual','Intervalo_de_Confianza_Error_Absoluto_Minimo','Intervalo_de_Confianza_Error_Absoluto_Maximo', 'Promedio_Error_Abs','Desviacion_Estandar_Abs', 'Intervalo_de_Confianza_Error_Porcentual_Minimo', 'Intervalo_de_Confianza_Error_Porcentual_Maximo']])
    latex2 = metricasRendimiento[['Metaheuristica','Promedio_Error_Porcentual', 'Desviacion_Estandar_Porcentual','Intervalo_de_Confianza_Error_Absoluto_Minimo','Intervalo_de_Confianza_Error_Absoluto_Maximo', 'Promedio_Error_Abs','Desviacion_Estandar_Abs', 'Intervalo_de_Confianza_Error_Porcentual_Minimo', 'Intervalo_de_Confianza_Error_Porcentual_Maximo']].to_latex(index=False, column_format='lrrrrrrrrr', caption=f'Métricas de rendimiento {pipeline}.', label=f'tab:{pipeline}_metricas' )
    ### Revisar aqui!!!! Este es donde se rompe el codigo.
    desempeñoPorSolver = resultadosFiltrados.groupby('Metaheuristica')['Mascara Exito'].agg(
        TotalExperimentos='count',
        TotalExitos='sum'
    ).reset_index()
    print("--- Desempeño por Solver ---")
    print(desempeñoPorSolver)
    ###AQUI HAY ERROR. Corregir. Tiene que ser to total experimentos - TOTALEXITOSxSOLVER.
    desempeñoPorSolver['TotalFallos'] = desempeñoPorSolver['TotalExperimentos'] - desempeñoPorSolver['TotalExitos']
    desempeñoPorSolver['Tasa de Fallo'] = desempeñoPorSolver['TotalFallos'] / desempeñoPorSolver['TotalExperimentos']
    print("--- Tasa De Fallo por solver ---")
    print(desempeñoPorSolver[['Metaheuristica', 'TotalExperimentos', 'Tasa de Fallo']].sort_values(by='Tasa de Fallo'))
    latex3 = desempeñoPorSolver[['Metaheuristica', 'TotalExperimentos', 'Tasa de Fallo']].sort_values(by='Tasa de Fallo').to_latex(index=False, column_format='lrrr', caption=f'Tasa de fallo del procesado {pipeline}.', label=f'tab:{pipeline}_fallos' )
    Plotter.graficos_por_pipeline(FallosTot, desempeñoPorSolver, metricasRendimiento, totalExperimentos, pipeline)
    with open(output, 'a', encoding='utf-8') as f:
        ##tabla automagica en latex
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Resumen de fallos del procesado %s}\n" % pipeline)
        f.write("\\label{tab:%s_resumen_fallos}\n" % pipeline)
        f.write("\\begin{tabular}{lr}\n")
        f.write("\\hline\n")
        f.write("Tipo de fallo & Cantidad \\\\\n")
        f.write("\\hline\n")
        for key, count in FallosTot.items():
            if key not in ['Total Fallos', 'Total Exitos']:
                f.write(f"{key.replace('_',' ')} & {count} \\\\\n")
        f.write("\\hline\n")
        f.write(f"Tasa de Fallos & {tasaDeFallos:.2%} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        f.write("----------------------\n\n")
        
        f.write(latex1);f.write('----------------------\n\n')
        f.write(latex2);f.write('----------------------\n\n')
        f.write(latex3)
    return FallosTot, dfProcesado, desempeñoPorSolver, metricasRendimiento

def parsearResultado(resultado, Fallos):
    if isinstance(resultado, str):
        Fallos['Total Fallos'] += 1
        texto = resultado.lower()
        if 'failed to evaluate' in texto:
            Fallos['Failure_to_evaluate'] += 1
        elif 'failed to run target heuristic' in texto:
            Fallos['Failure_to_run_target_heuristic'] += 1
        elif 'failed to load' in texto:
            Fallos['Failure_to_load'] += 1
        else:
            Fallos['Otros'] += 1
        return np.nan

    if isinstance(resultado, tuple):
        resultado = list(resultado)
    elif isinstance(resultado, np.ndarray):
        resultado = resultado.tolist()

    if isinstance(resultado, list):

        for elem in resultado:
            if isinstance(elem, dict):
                if 'bestScore' in elem:
                    score = elem['bestScore']
                elif 'currentScore' in elem:
                    score = elem['currentScore']
                else:
                    continue

                if isinstance(score, numbers.Real):
                    Fallos['Total Exitos'] += 1
                    return float(score)
        numeric_vals = [e for e in resultado if isinstance(e, numbers.Real)]
        if numeric_vals:
            final_score = float(numeric_vals[-1]) 

            # Los resultados optimos rondan a los mas en los 3 digitos. Cualquier cosa superior a eso puede ser una penalización o bien producto de un error de evaluacion en el codigo generado
            # Un ejemplo claro de esto son las intancias n33 a n35 de GC. Estas tienen un multiplicador the 1000 por ningun buen motivo
            # Generalmente es preferible no moverse a posiciones inviables, asi que sea como sea, es un fallo de ejecución. 
            if abs(final_score) >= 1e3:                 
                Fallos['Total Fallos'] += 1
                Fallos['Otros'] += 1
                return np.nan

            Fallos['Total Exitos'] += 1
            return final_score

        Fallos['Total Fallos'] += 1
        Fallos['Otros'] += 1
        return np.nan

    Fallos['Total Fallos'] += 1
    Fallos['Otros'] += 1
    return np.nan

#Porque ciertos resultados son rarificos, debido a las halucinaciones
def normalizar_resultado(x):
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        x = x.strip()
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(x)
        except (SyntaxError, ValueError):
            return x
    return x

# nunca se uso
def probarCorrectitud(MejorConocida, valorMejorConocida, componentes, resultado):
    try:
        evaluacion = cargarComponente(componentes['EVAL_CODE'])
        print(f"Loaded 'EVAL_CODE' into variable 'evaluate_solution'. Name: {eval.__name__}")
    except Exception as e:
        return(f"Failed to load EVAL_CODE: {e}")
    try:
        vecindad = cargarComponente(componentes['NB_CODE'])
        print(f"Loaded 'NB_CODE' into variable 'generate_neighbour'. Name: {vecindad.__name__}")
    except Exception as e:
        return(f"Failed to load NB_CODE: {e}")
    
    objetivoErronea = 0
    if(evaluacion(MejorConocida) != valorMejorConocida):
        objetivoErronea = 1
    diffConMejor = valorMejorConocida - evaluacion(resultado)
    return

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