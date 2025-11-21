import ast
import argparse
import numbers
import sys
import time
from dotenv import load_dotenv
import pandas as pd
import Plotter
import numpy as np
import random
import math
from scipy.stats import t, fisher_exact
import os
import Optimizacion.PromptSamplerOP as PromptSamplerOP
import json
from typing import List, Union,Callable, Any, Dict
from DefinicionMatematica import DefinicionMat
from Instancias import DataLoader
from Generador import generador
import Heuristicas.IteratedLocalSearch
import Heuristicas.SimulatedAnnealing
import Heuristicas.TabooSearch

def main():
    semilla = 1234
    iteraciones = 3 #Genracion se atasca muy seguido. Se prueba sin reiniciar, luego probabos con reinicio. 
    filaInicio = 3
    filaMultiplo = 3
    #Inicializar datos
    load_dotenv()
    #Modificacion para pruebas, prepara modo batch por defecto. Estas lineas se tienen que eliminar cuando se empieze a optimizar
    if len(sys.argv) == 1:
        sys.argv.append('-o')
    pathDB= os.path.join(os.path.dirname(__file__), 'Data')
    #Fin modificacion para pruebas
    os.makedirs(pathDB, exist_ok=True)
    componentesPath = os.path.join(pathDB, 'componentes-SR-GC.jsonl')
    problemasPath = os.path.join(pathDB, 'problemas-GC.jsonl')
    feedbackPath = os.path.join(pathDB,'feedback-SR-GC.jsonl')
    resultPath = os.path.join(pathDB,'resultados-SR-GC.jsonl')
    instancias = DataLoader()
    instancias.cargarProblemas()
    llms = generador()
    llms.cargarLLMs()
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch',action='store_true', dest='batch_def', help='Definir problemas en batch, no optimiza')
    parser.add_argument('-o', '--opt',action='store_true', dest='skip_def', help='Solo optimizar, no define')
    parser.add_argument('-n', '--nd',action='store_true', dest='not_def', help='Optimizar pero no define')
    parser.add_argument('-p', '--plot',action='store_true', dest='plot', help='Procesar resultados')
    parser.add_argument('problem_ID', nargs='?',default=None,help="ID del problema a optimizar: Formato: Tipo_dataset_ID")
    args=parser.parse_args()

    if args.plot:
        problemaDB = pd.read_json(problemasPath,lines=True)
        if os.path.exists(componentesPath) and os.path.exists(resultPath) and os.path.exists(feedbackPath):
            resultDB = cargarResultados(os.path.join(pathDB,'resultados-SR-K.jsonl'))
            resultControlDB = cargarResultados(os.path.join(pathDB,'resultados-SR-K-SD.jsonl'))
            resultUDDB = cargarResultados(os.path.join(pathDB,'resultados-SR-K-PR.jsonl'))
            resultGCDB = cargarResultados(os.path.join(pathDB,'resultados-SR-GC.jsonl'))

            output = os.path.join(pathDB,'tablasLatex.tex')
            fallosFP, dfProcesadoFP, desempeñoPorSolverFP, metricasRendimientoFP = procesarResultados(resultDB, output,"CD")
            fallosC, dfProcesadoC, desempeñoPorSolverC, metricasRendimientoC = procesarResultados(resultControlDB, output,"SD")
            fallosC, dfProcesadoC, desempeñoPorSolverU, metricasRendimientoU = procesarResultados(resultUDDB, output,"UD")
            fallosC, dfProcesadoC, desempeñoPorSolverU, metricasRendimientoU = procesarResultados(resultGCDB, output,"GC")
            pd.set_option('display.float_format', lambda x: '%.0000f' % x) #Poco elegante pero funciona
            comparacion = metricasRendimientoFP.merge(
                metricasRendimientoC,
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
            fisher_df = desempeñoPorSolverFP.merge(desempeñoPorSolverC,on='Metaheuristica',suffixes=('_FP', '_Control'))
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
        else:
            print("No hay resultados disponibles. Corra el algoritmo primero")
        return 0

    if args.skip_def:
        problemaDB = pd.read_json(problemasPath,lines=True)
        if os.path.exists(componentesPath) and os.path.exists(resultPath) and os.path.exists(feedbackPath):
            componenteDB = pd.read_json(componentesPath,lines=True)
            feedbackDB = pd.read_json(feedbackPath,lines=True)
            resultDB = pd.read_json(resultPath,lines=True)
        else:
            componenteDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Evaluacion', 'Vecindad', 'Perturbacion','Version'])
            feedbackDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Componente','Version', 'Feedback'])
            resultDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Evaluacion', 'Vecindad', 'Perturbacion', 'Resultados','Solucion','Valor Optimo', 'Metaheuristica', 'Tiempo'])
        problemasFiltrados = problemaDB.iloc[filaMultiplo::filaMultiplo]
        for tuplaProblema in problemasFiltrados.itertuples(index=False):    
            componenteDB, feedbackDB, resultDB = optimizarProblemaPredefinido(tuplaProblema, componenteDB,resultDB,feedbackDB, iteraciones)
            componenteDB.to_json(componentesPath,orient='records',lines=True)
            feedbackDB.to_json(feedbackPath,orient='records',lines=True)
            resultDB.to_json(resultPath,orient='records',lines=True)
        return 0
    #Problemas en batch
    if args.batch_def:
        print("Datos inicializados. Iniciando definicion matematica")
        DefinicionMat.definirBatch(instancias,llms, problemasPath, tipo="graph_coloring")
        return 0
    if args.not_def:
        print("Datos inicializados. Iniciando definicion matematica")
        problemaDB = pd.read_json(problemasPath,lines=True)
        if os.path.exists(componentesPath) and os.path.exists(resultPath) and os.path.exists(feedbackPath):
            componenteDB = pd.read_json(componentesPath,lines=True)
            feedbackDB = pd.read_json(feedbackPath,lines=True)
            resultDB = pd.read_json(resultPath,lines=True)
        else:
            componenteDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Evaluacion', 'Vecindad', 'Perturbacion','Version'])
            feedbackDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Componente','Version', 'Feedback'])
            resultDB = pd.DataFrame(columns=['ID_Problema', 'Representacion', 'Evaluacion', 'Vecindad', 'Perturbacion', 'Resultados','Solucion','Valor Optimo', 'Metaheuristica', 'Tiempo'])
        problemasFiltrados = problemaDB.iloc[filaInicio::filaMultiplo]
        for tuplaProblema in problemasFiltrados.itertuples(index=False):    
            componenteDB, feedbackDB, resultDB = optimizarProblemaEnBruto(tuplaProblema, componenteDB,resultDB,feedbackDB, iteraciones)
            componenteDB.to_json(componentesPath,orient='records',lines=True)
            feedbackDB.to_json(feedbackPath,orient='records',lines=True)
            resultDB.to_json(resultPath,orient='records',lines=True)
        return 0
    else:
    #Problema individual
        respuesta,feedback,resultados = DefinicionMat.definirIndividual(instancias,llms,problemasPath,"graph_coloring_random_dataset_in_house_9_8")
    #componenteDB = pd.read_json(componentesPath,lines=True)
    problemaDB = pd.read_json(problemasPath,lines=True)
    #feedbackDB = pd.read_json(feedbackPath,lines=True)
    #resultDB = pd.read_json(resultPath,lines=True)
    #componenteDB.to_json(componentesPath,lines=True)
    #feedbackDB.to_json(feedbackPath,lines=True)
    #resultDB.to_json(resultPath,lines=True)
    return 0

#Utiliza las DB para guardar las respuestas con versionado. Eso significa que hay que acceder a los archivos y escribir sobre ellos. 
#Podemos pasar el problema definido en la fase 1 directamente, en vez de sacar uno de la DB. Eso nos deja con dos variantes: seed, y problema predefinido. En ambos casos resultDB, feedbackDB y componenteDB son necesarios
#actualmente solo esta evaluando 1 componente a la vez, tiene que evaluar sets de componentes. Representacion, Vecindad y Evaluacion
def optimizarProblemaAleatorio(problemaDB:pd.DataFrame,componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame,semilla, iteraciones):
        rawProblema = PromptSamplerOP.sampleProblemaDB(problemaDB,semilla)
        componenteDB, feedbackDB, resultDB = optimizarProblemaPredefinido(rawProblema, componenteDB, resultDB, feedbackDB, iteraciones)
        return componenteDB, feedbackDB, resultDB

def optimizarProblemaPredefinido(problema,componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame, iteraciones):
        problemaID, defProblema, solucion, objetivo, seedPrompt = PromptSamplerOP.generateSeedPrompt(problema)
        print(problemaID, solucion, objetivo)
        llms = generador()
        llms.cargarLLMs()
        respuestaInicial = llms.generarComponentes(seedPrompt)
        textoBruto = respuestaInicial.content[0]['text']
        componentes = json.loads(textoBruto)
        print(componentes)
        respuestas = [] #Temporal, para gguardar todas las respuestas de generacion de componentes para tener la metadata.
        feedbacks = []
        i = 0
        while i < iteraciones:
            resultadoSA, tiempoSA = cronometrarFuncion(evaluarComponentesSAcd ,componentes)
            resultadoILS, tiempoILS = cronometrarFuncion(evaluarComponentesILScd, componentes)
            resultadoTS, tiempoTS = cronometrarFuncion(evaluarComponentesTScd, componentes)
            print(resultadoSA)
            print(resultadoILS)
            print(resultadoTS)
            resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoSA,solucion, objetivo, "SA", tiempoSA)
            resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoILS,solucion, objetivo, "ILS", tiempoILS)
            resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoTS,solucion, objetivo, "TS", tiempoTS)
            feedbackPrompt = PromptSamplerOP.generateFeedbackPrompt(defProblema,componentes,resultadoSA, resultadoILS,resultadoTS, solucion, objetivo) #Necesita trabajar con el nuevo sistema de JSON
            feedback = llms.generarFeedback(feedbackPrompt) 
            feedbacks.append(feedback)
            feedbackTexto = feedback.content[0]['text']
            print(feedbackTexto)
            feedbackDB = guardarFeedback(problemaID,feedbackDB,componentes['REPRESENTATION'], componentes, i,feedbackTexto)
            newPrompt = PromptSamplerOP.updatePromptOS(defProblema,componentes,resultadoSA,feedbackTexto)
            print(newPrompt)
            respuesta = llms.generarComponentes(newPrompt)
            respuestas.append(respuesta)
            textoBruto = respuesta.content[0]['text']
            print(textoBruto)
            componentes = json.loads(textoBruto)
            componenteDB = guardarComponentes(problemaID,componenteDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'], i)
            i = i + 1
            if i % 3 == 0:
                llms = DefinicionMat.reiniciarLLMSDef()
                print("Maquinas re-instanciadas")
        resultadoSA, tiempoSA = cronometrarFuncion(evaluarComponentesSAcd,componentes)
        resultadoILS, tiempoILS = cronometrarFuncion(evaluarComponentesILScd, componentes)
        resultadoTS, tiempoTS = cronometrarFuncion(evaluarComponentesTScd, componentes)
        print(resultadoSA)
        print(resultadoILS)
        print(resultadoTS)
        resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoSA,solucion, objetivo, "SA", tiempoSA)
        resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoILS,solucion, objetivo, "ILS", tiempoILS)
        resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoTS,solucion, objetivo, "TS", tiempoTS)        
        return componenteDB, feedbackDB, resultDB

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

def procesarResultados(resultados:pd.DataFrame, output, pipeline):
    pd.set_option('display.float_format', lambda x: '%.0f' % x)
    resultados['Resultados'] = resultados['Resultados'].apply(normalizar_resultado) #Convierte el resultado en algo parseable
    Fallos = {
    'Failure_to_evaluate': 0,
    'Failure_to_run_target_heuristic': 0,
    'Failure_to_load': 0,
    'Otros': 0,
    'Total Fallos': 0,
    'Total Exitos': 0
    }
    resultadosFiltrados = resultados[~resultados['ID_Problema'].str.contains('inverted', case=False, na=False)].copy() #Para compensar por error en parser de soluciones. Se eliminan los problemas invertidos
    resultadosFiltrados['Puntaje Real'] = resultadosFiltrados['Resultados'].apply(lambda r: parsearResultado(r, Fallos))
    print(resultadosFiltrados['Puntaje Real'])
    dfProcesado = resultadosFiltrados[resultadosFiltrados['Puntaje Real'].notna()].copy()
    print(dfProcesado[['Metaheuristica', 'Resultados', 'Puntaje Real', 'Valor Optimo']])
    latex1 = dfProcesado[['Metaheuristica', 'Resultados', 'Puntaje Real', 'Valor Optimo']].to_latex(index=False, column_format='lrrrr', caption=f'Resultados del procesado {pipeline}.', label=f'tab:{pipeline}_resultados_generales' )
    for key, count in Fallos.items():
        if key not in ['Total Fallos', 'Total Exitos']:
            print(f"* {key}: {count}")
    totalExperimentos =  len(resultadosFiltrados)
    tasaDeFallos = Fallos['Total Fallos'] / totalExperimentos
    print(f"Tasa de Fallos {tasaDeFallos:.2%}")
    dfProcesado['Diferencia Absoluta'] = (-dfProcesado['Puntaje Real']-dfProcesado['Valor Optimo']).abs() #En base a revision manual, el Puntaje real esta siempre cambiado de signo, pero colapsa a la respuesta optima igual. Se cambio el signo del puntaje real a negativo para tomar en cuenta esto
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
    desempeñoPorSolver = resultadosFiltrados.groupby('Metaheuristica')['Mascara Exito'].agg(
        TotalExperimentos='count',
        TotalExitos='sum'
    ).reset_index()
    print("--- Desempeño por Solver ---")
    print(desempeñoPorSolver)
    desempeñoPorSolver['TotalFallos'] = desempeñoPorSolver['TotalExperimentos'] - desempeñoPorSolver['TotalExitos']
    desempeñoPorSolver['Tasa de Fallo'] = desempeñoPorSolver['TotalFallos'] / desempeñoPorSolver['TotalExperimentos']
    print("--- Tasa De Fallo por solver ---")
    print(desempeñoPorSolver[['Metaheuristica', 'TotalExperimentos', 'Tasa de Fallo']].sort_values(by='Tasa de Fallo'))
    latex3 = desempeñoPorSolver[['Metaheuristica', 'TotalExperimentos', 'Tasa de Fallo']].sort_values(by='Tasa de Fallo').to_latex(index=False, column_format='lrrr', caption=f'Tasa de fallo del procesado {pipeline}.', label=f'tab:{pipeline}_fallos' )
    Plotter.graficos_por_pipeline(Fallos, desempeñoPorSolver, metricasRendimiento, pipeline)
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
        for key, count in Fallos.items():
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
    return Fallos, dfProcesado, desempeñoPorSolver, metricasRendimiento

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

        # 2.a) Si hay un dict con bestScore / currentScore, usamos eso
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

        # Si no hay dict relevante, buscamos números en la lista
        numeric_vals = [e for e in resultado if isinstance(e, numbers.Real)]
        if numeric_vals:
            final_score = float(numeric_vals[-1])  # usar el ÚLTIMO número (p.ej. 223.0, -59.0, etc.)

            # Regla opcional: filtrar penalties absurdos
            if abs(final_score) > 1e8:  # ajusta el umbral si quieres
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


def optimizarProblemaEnBruto(problema,componenteDB:pd.DataFrame,resultDB: pd.DataFrame,feedbackDB: pd.DataFrame, iteraciones):
        problemaID, defProblema, solucion, objetivo, seedPrompt = PromptSamplerOP.generateRawSeedPrompt(problema)
        print(problemaID, solucion, objetivo)
        llms = generador()
        llms.cargarLLMs()
        respuestaInicial = llms.generarComponentes(seedPrompt)
        textoBruto = respuestaInicial.content[0]['text']
        componentes = json.loads(textoBruto)
        print(componentes)
        respuestas = [] #Temporal, para gguardar todas las respuestas de generacion de componentes para tener la metadata.
        feedbacks = []
        i = 0
        while i < iteraciones:
            resultadoSA, tiempoSA = cronometrarFuncion(evaluarComponentesSAud ,componentes)
            resultadoILS, tiempoILS = cronometrarFuncion(evaluarComponentesILSud, componentes)
            resultadoTS, tiempoTS = cronometrarFuncion(evaluarComponentesTSud, componentes)
            print(resultadoSA)
            print(resultadoILS)
            print(resultadoTS)
            resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoSA,solucion, objetivo, "SA", tiempoSA)
            resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoILS,solucion, objetivo, "ILS", tiempoILS)
            resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoTS,solucion, objetivo, "TS", tiempoTS)
            feedbackPrompt = PromptSamplerOP.generateFeedbackPrompt(defProblema,componentes,resultadoSA, resultadoILS,resultadoTS, solucion, objetivo) #Necesita trabajar con el nuevo sistema de JSON
            feedback = llms.generarFeedback(feedbackPrompt) 
            feedbacks.append(feedback)
            feedbackTexto = feedback.content[0]['text']
            print(feedbackTexto)
            feedbackDB = guardarFeedback(problemaID,feedbackDB,componentes['REPRESENTATION'], componentes, i,feedbackTexto)
            newPrompt = PromptSamplerOP.updatePromptOS(defProblema,componentes,resultadoSA,feedbackTexto)
            print(newPrompt)
            respuesta = llms.generarComponentes(newPrompt)
            respuestas.append(respuesta)
            textoBruto = respuesta.content[0]['text']
            print(textoBruto)
            componentes = json.loads(textoBruto)
            componenteDB = guardarComponentes(problemaID,componenteDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'], i)
            i = i + 1
            if i % 3 == 0:
                llms = DefinicionMat.reiniciarLLMSDef()
                print("Maquinas re-instanciadas")
        resultadoSA, tiempoSA = cronometrarFuncion(evaluarComponentesSAud,componentes)
        resultadoILS, tiempoILS = cronometrarFuncion(evaluarComponentesILSud, componentes)
        resultadoTS, tiempoTS = cronometrarFuncion(evaluarComponentesTSud, componentes)
        print(resultadoSA)
        print(resultadoILS)
        print(resultadoTS)
        resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoSA,solucion, objetivo, "SA", tiempoSA)
        resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoILS,solucion, objetivo, "ILS", tiempoILS)
        resultDB = guardarResultado(problemaID,resultDB, componentes['REPRESENTATION'],componentes['EVAL_CODE'],componentes['NB_CODE'],componentes['PERTURB_CODE'],resultadoTS,solucion, objetivo, "TS", tiempoTS)        
        return componenteDB, feedbackDB, resultDB

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

def evaluarComponentesILScd(componentes):
    #Guardar excepciones como string, para retornarlas como debug
    try:
        solucionPrueba = ast.literal_eval(componentes['SAMPLE_SOL'].strip().replace(' ', ''))
    except Exception as e:
        return(f"Failed to load SAMPLE_SOL: {e}")
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
    try:
        perturb = cargarComponente(componentes['PERTURB_CODE'])
        print(f"Loaded 'PERTURB_CODE' into variable 'perturb_solution'. Name: {perturb.__name__}")
    except Exception as e:
        return(f"Failed to load PERTURB_CODE: {e}")
    try:
        valor = evaluacion(solucionPrueba)
    except Exception as e:
        return(f"Failed to evaluate SAMPLE_SOL with EVAL_CODE: {e}")
    try:
        resultadoILS = Heuristicas.SimulatedAnnealing.SA(solucionPrueba,solucionPrueba,valor,vecindad,evaluacion,1000,10,0.9)
    except Exception as e:
        return(f"Failed to run target heuristic: {e}.  Signature def SA(solution,best_sol, best_score, generate_neighbour(), evaluate_solution(), TEMP, MIN_TEMP, cooling_factor)")
    #Cargar heuristicas, retornar resultados de cada una.
    return resultadoILS

def evaluarComponentesSAcd(componentes):
    #Guardar excepciones como string, para retornarlas como debug
    try:
        solucionPrueba = ast.literal_eval(componentes['SAMPLE_SOL'].strip().replace(' ', ''))
    except Exception as e:
        return(f"Failed to load SAMPLE_SOL: {e}")
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
    try:
        perturb = cargarComponente(componentes['PERTURB_CODE'])
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

def evaluarComponentesTScd(componentes):
    #Guardar excepciones como string, para retornarlas como debug
    try:
        solucionPrueba = ast.literal_eval(componentes['SAMPLE_SOL'].strip().replace(' ', ''))
    except Exception as e:
        return(f"Failed to load SAMPLE_SOL: {e}")
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
    try:
        perturb = cargarComponente(componentes['PERTURB_CODE'])
        print(f"Loaded 'PERTURB_CODE' into variable 'perturb_solution'. Name: {perturb.__name__}")
    except Exception as e:
        return(f"Failed to load PERTURB_CODE: {e}")
    try:
        valor = evaluacion(solucionPrueba)
    except Exception as e:
        return(f"Failed to evaluate SAMPLE_SOL with EVAL_CODE: {e}")
    try:
        resultadoTS = Heuristicas.SimulatedAnnealing.SA(solucionPrueba,solucionPrueba,valor,vecindad,evaluacion,1000,10,0.9)
    except Exception as e:
        return(f"Failed to run target heuristic: {e}.  Signature def SA(solution,best_sol, best_score, generate_neighbour(), evaluate_solution(), TEMP, MIN_TEMP, cooling_factor)")
    #Cargar heuristicas, retornar resultados de cada una.
    return resultadoTS

def evaluarComponentesSAud(componentes):
    #Guardar excepciones como string, para retornarlas como debug
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
    try:
        solucionPrueba = componentes['SAMPLE_SOL']
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

def evaluarComponentesILSud(componentes):
    #Guardar excepciones como string, para retornarlas como debug
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
    try:
        perturb = cargarComponente(componentes['PERTURB_CODE'])
        print(f"Loaded 'PERTURB_CODE' into variable 'perturb_solution'. Name: {perturb.__name__}")
    except Exception as e:
        return(f"Failed to load PERTURB_CODE: {e}")
    try:
        solucionPrueba = componentes['SAMPLE_SOL']
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

def evaluarComponentesTSud(componentes):
    #Guardar excepciones como string, para retornarlas como debug
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
    try:
        perturb = cargarComponente(componentes['PERTURB_CODE'])
        print(f"Loaded 'PERTURB_CODE' into variable 'perturb_solution'. Name: {perturb.__name__}")
    except Exception as e:
        return(f"Failed to load PERTURB_CODE: {e}")
    try:
        solucionPrueba = componentes['SAMPLE_SOL']
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

def cronometrarFuncion(func: Callable, *args, **kwargs) -> tuple[float, any]:
    inicio = time.perf_counter()
    resultado = func(*args, **kwargs)
    fin = time.perf_counter()
    tiempo = fin - inicio
    return resultado, tiempo


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
def guardarFeedback(problemaID,feedbackDB, representacion, componentes, version, feedback):
    datosFeedback = {
        'ID_Problema': problemaID,
        'Representacion': representacion,
        'Componentes': componentes,
        'Version': version,
        'Feedback': feedback
    }
    dfAux = pd.DataFrame([datosFeedback])
    feedbackDBMod = pd.concat([feedbackDB,dfAux], ignore_index=True)
    return feedbackDBMod
def guardarComponentes(problemaID,componenteDB, representacion, evaluacion, vecindad, perturbacion, version):
    datosComponentes = {
        'ID_Problema': problemaID,
        'Representacion': representacion,
        'Evaluacion': evaluacion,
        'Vecindad': vecindad,
        'Perturbacion': perturbacion,
        'Version': version
    }
    dfAux = pd.DataFrame([datosComponentes])
    componenteDBMod = pd.concat([componenteDB,dfAux], ignore_index=True)
    return componenteDBMod

main()