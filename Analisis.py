from dataclasses import dataclass
from enum import Enum, auto
import json
import pandas as pd
import numpy as np
import ast
import numbers
from Instancias import DataLoader
from scipy.stats import t

import Plotter

#Esto causara un fallo immediato si se cambian los outputs del evaluador sin actualizar _clasificarFallo tambien.
class TipoFallo(Enum):
    FAILURE_TO_EVALUATE = auto()
    FAILURE_TO_RUN_TARGET_HEURISTIC = auto()
    FAILURE_TO_LOAD = auto()
    SCORE_INVALIDO = auto()  # Se assume que inf equivale a >=1e3 debido al rango de puntajes promedio. Con otros problemas esta restriccion se tiene que relajar 
    OTRO = auto()

@dataclass
class ResultadoParseado:
    score: float | None
    fallo: TipoFallo | None  

    @property
    def valido(self) -> bool:
        return self.score is not None and self.fallo is None

def _getScore(id_problema: str,instancias: DataLoader):
    inst = instancias.getInstancia(id_problema)
    return inst.objectiveScore if inst is not None else None

def _getSolucion(id_problema: str,instancias: DataLoader):
    inst = instancias.getInstancia(id_problema)
    return inst.parsedSolution if inst is not None else None

def _normalizarResultado(resultado):
    if isinstance(resultado, (list, dict)):
        return resultado
    if isinstance(resultado, str):
        resultado = resultado.strip()
        try:
            return json.loads(resultado)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(resultado)
        except (SyntaxError, ValueError):                
            return resultado
    return resultado

def _contarCorrectitud(comparaciones: pd.Series) -> dict:
    return {
        'Soluciones Identicas': int(comparaciones.sum()),
        'Soluciones Distintas': int((~comparaciones).sum()),
    }

def _clasificarFallo(texto: str) -> TipoFallo:
    texto = texto.lower()
    if 'failed_to_evaluate' in texto:
        return TipoFallo.FAILURE_TO_EVALUATE
    if 'failed_to_run_target_heuristic' in texto:
        return TipoFallo.FAILURE_TO_RUN_TARGET_HEURISTIC
    if 'failed_to_load' in texto:
        return TipoFallo.FAILURE_TO_LOAD
    return TipoFallo.OTRO

def _extraerPuntaje(resultado) -> float | None:
    if isinstance(resultado, tuple):
        resultado = list(resultado)
    elif isinstance(resultado, np.ndarray):
        resultado = resultado.tolist()

    if not isinstance(resultado, list):
        return None

    for elem in resultado:
        if isinstance(elem, dict):
            score = elem.get('bestScore') or elem.get('currentScore')
            if isinstance(score, numbers.Real):
                return float(score)

    # En caso de que no se encuentre el la columna correcta, se hace fallback al penultimo valor numerico obtenido
    numeric_vals = [e for e in resultado if isinstance(e, numbers.Real)]
    if numeric_vals:
        return float(numeric_vals[-1])

    return None

def parsearResultado(resultado) -> ResultadoParseado:
    if isinstance(resultado, str):
        return ResultadoParseado(score=None, fallo=_clasificarFallo(resultado))

    score = _extraerPuntaje(resultado)

    if score is None:
        return ResultadoParseado(score=None, fallo=TipoFallo.OTRO)

    if abs(score) >= 1e3:
        return ResultadoParseado(score=None, fallo=TipoFallo.SCORE_INVALIDO)

    return ResultadoParseado(score=score, fallo=None)

def _contarFallos(resultadosParseados: pd.Series) -> dict:
    fallos = {t: 0 for t in TipoFallo}
    exitos = 0

    for resultado in resultadosParseados:
        if resultado.valido:
            exitos += 1
        else:
            fallos[resultado.fallo] += 1

    return {
        'Failure_to_evaluate':             fallos[TipoFallo.FAILURE_TO_EVALUATE],
        'Failure_to_run_target_heuristic': fallos[TipoFallo.FAILURE_TO_RUN_TARGET_HEURISTIC],
        'Failure_to_load':                 fallos[TipoFallo.FAILURE_TO_LOAD],
        'Invalid_score':                  fallos[TipoFallo.SCORE_INVALIDO],
        'Otros':                           fallos[TipoFallo.OTRO],
        'Total Fallos':                    sum(fallos.values()),
        'Total Exitos':                    exitos,
    }

def _compararSoluciones(solucionGenerada, solucionInstancia) -> bool:
    try:
        final_solution = solucionGenerada[2]
        return list(final_solution) == list(solucionInstancia)
    except (TypeError, IndexError):
        return False


def procesarResultados(resultados: pd.DataFrame, instancias: DataLoader):
    pd.set_option('display.float_format', lambda x: '%.0f' % x)
    resultadosAux = resultados.copy()
    resultadosAux['Valor Optimo'] = resultadosAux['ID_Problema'].map(
        lambda id_: _getScore(id_, instancias)
    )
    resultadosAux['Solucion'] = resultadosAux['ID_Problema'].map(
        lambda id_: _getSolucion(id_, instancias)
    )
    resultadosAux['Resultados'] = resultadosAux['Resultados'].apply(_normalizarResultado)

    resultadosParseados = resultadosAux['Resultados'].apply(parsearResultado)

    puntajes = resultadosParseados.apply(lambda r: r.score if r.valido else np.nan)

    resultadosAux['Puntaje Real'] = puntajes  

    dfProcesado = resultadosAux[puntajes.notna()].copy()  

    dfProcesado['Solucion identica'] = dfProcesado.apply(
        lambda row: _compararSoluciones(row['Resultados'], row['Solucion']),
        axis=1
    )

    
    dfProcesado['Solucion identica'] = dfProcesado.apply(
        lambda row: _compararSoluciones(row['Resultados'], row['Solucion']), axis=1
    )

    tiposFallos = _contarFallos(resultadosParseados)
    
    correctitud = _contarCorrectitud(dfProcesado['Solucion identica'])

    dfProcesado['Diferencia Absoluta']    = ( dfProcesado['Puntaje Real'] -  dfProcesado['Valor Optimo']).abs()
    dfProcesado['Diferencia Porcentual']  = ( dfProcesado['Diferencia Absoluta'] /  dfProcesado['Valor Optimo']).abs() * 100 #Valor optimo de EHOP nunca es cero

    resultadosAux['TipoFallo'] = resultadosParseados.apply(lambda r: r.fallo.name if not r.valido else None)

    return dfProcesado,tiposFallos, resultadosAux,  correctitud

def agregarResultados(dfProcesado: pd.DataFrame, resultadosAux: pd.DataFrame) -> dict:
    fallos      = len(resultadosAux) - len(dfProcesado)
    ejecuciones = int((~dfProcesado['Solucion identica']).sum())
    optimos     = int(dfProcesado['Solucion identica'].sum())
    assert fallos + ejecuciones + optimos == len(resultadosAux), "Los totales no cuadran"
    return {'Fallos': fallos, 'Ejecuciones': ejecuciones, 'Optimos': optimos}

def generarFigurasYTablasLatexLocales(dfProcesado: pd.DataFrame, FallosTot: dict, resultadosAux: pd.DataFrame, correctitud: dict, output: str, pipeline: str):
    totalExperimentos = len(resultadosAux)
    tasaDeFallos      = FallosTot['Total Fallos'] / totalExperimentos
    metricasRendimiento = _calcularMetricasRendimiento(dfProcesado)
    desempeñoPorSolver  = _desempeñoPorSolver(resultadosAux)

    _imprimirResultados(FallosTot, tasaDeFallos, dfProcesado, correctitud, metricasRendimiento, desempeñoPorSolver)
    _crearTablasLatex(dfProcesado, metricasRendimiento, desempeñoPorSolver, FallosTot, tasaDeFallos, output, pipeline)
    _plots(totalExperimentos, pipeline, dfProcesado, resultadosAux)

    return FallosTot, desempeñoPorSolver, metricasRendimiento

def _calcularMetricasRendimiento(dfProcesado):
    ##  metricas rendimiento
    metricasRendimiento = dfProcesado.groupby('Metaheuristica').agg(
        exitosTotales = ('Puntaje Real', 'count'),
        Promedio_Error_Abs =('Diferencia Absoluta', 'mean'),
        Desviacion_Estandar_Abs = ('Diferencia Absoluta', 'std'),
        Promedio_Error_Porcentual =('Diferencia Porcentual', 'mean'),
        Desviacion_Estandar_Porcentual = ('Diferencia Porcentual', 'std')
    ).reset_index()
    alpha = 0.05
    tCritico = t.ppf(1-alpha/2, metricasRendimiento['exitosTotales']-1)
    margenAbs = metricasRendimiento['Desviacion_Estandar_Abs']/np.sqrt(metricasRendimiento['exitosTotales'])
    margenPor = metricasRendimiento['Desviacion_Estandar_Porcentual']/np.sqrt(metricasRendimiento['exitosTotales'])
    #Para saber que tan generlizables son los resultados, considerando la pequeña muestra
    metricasRendimiento['Intervalo_de_Confianza_Error_Absoluto_Minimo'] = ( metricasRendimiento['Promedio_Error_Abs'] - tCritico * margenAbs)
    metricasRendimiento['Intervalo_de_Confianza_Error_Absoluto_Maximo'] = ( metricasRendimiento['Promedio_Error_Abs'] + tCritico * margenAbs)
    metricasRendimiento['Intervalo_de_Confianza_Error_Porcentual_Minimo'] = ( metricasRendimiento['Promedio_Error_Porcentual'] - tCritico * margenPor )
    metricasRendimiento['Intervalo_de_Confianza_Error_Porcentual_Maximo'] = ( metricasRendimiento['Promedio_Error_Porcentual'] + tCritico * margenPor )
    return metricasRendimiento

def _desempeñoPorSolver(resultadosAux: pd.DataFrame):
    desempeñoPorSolver = (resultadosAux
        .assign(Exito=resultadosAux['Puntaje Real'].notna()) ## crea una copia y le aplica una mascara para que filtremos todo lo que no sea una ejecucion exitosa
        .groupby('Metaheuristica')['Exito']
        .agg(TotalExperimentos='count', TotalExitos='sum')
        .reset_index()
    )
    desempeñoPorSolver['TotalFallos'] = desempeñoPorSolver['TotalExperimentos'] - desempeñoPorSolver['TotalExitos']
    desempeñoPorSolver['Tasa de Fallo'] = desempeñoPorSolver['TotalFallos'] / desempeñoPorSolver['TotalExperimentos']
    return desempeñoPorSolver

def _imprimirResultados(FallosTot, tasaDeFallos, dfProcesado, correctitud, metricasRendimiento, desempeñoPorSolver):
    for key, count in FallosTot.items():
        if key not in ['Total Fallos', 'Total Exitos']:
            print(f"* {key}: {count}")
    print("--- Metricas estandar de desempeño ---")
    print(metricasRendimiento[['Metaheuristica','Promedio_Error_Porcentual', 'Desviacion_Estandar_Porcentual','Intervalo_de_Confianza_Error_Absoluto_Minimo','Intervalo_de_Confianza_Error_Absoluto_Maximo', 'Promedio_Error_Abs','Desviacion_Estandar_Abs', 'Intervalo_de_Confianza_Error_Porcentual_Minimo', 'Intervalo_de_Confianza_Error_Porcentual_Maximo']])
    print(f"Tasa de Fallos {tasaDeFallos:.2%}")
    print(dfProcesado[['Metaheuristica', 'Resultados', 'Puntaje Real', 'Valor Optimo']])
    print("--- Desempeño por Solver ---")
    print(desempeñoPorSolver)
    totalExperimentosExitosos = len(dfProcesado)
    print("---Soluciones identicas al optimo---")
    print(f"Éxitos en la práctica: {correctitud['Soluciones Identicas']} / {totalExperimentosExitosos} "
          f"({correctitud['Soluciones Identicas']/totalExperimentosExitosos:.2%})")
    print(f"Fallos en la práctica: {correctitud['Soluciones Distintas']} / {totalExperimentosExitosos} "
          f"({correctitud['Soluciones Distintas']/totalExperimentosExitosos:.2%})")
    print("--- Tasa De Fallo por solver ---")
    print(desempeñoPorSolver[['Metaheuristica', 'TotalExperimentos', 'Tasa de Fallo']].sort_values(by='Tasa de Fallo'))

def _crearTablasLatex(dfProcesado, metricasRendimiento, desempeñoPorSolver, FallosTot, tasaDeFallos, output, pipeline):
    ## escribir latex
    latex1 = dfProcesado[['Metaheuristica', 'Resultados', 'Puntaje Real', 'Valor Optimo']].to_latex(index=False, column_format='lrrrr', caption=f'Resultados del procesado {pipeline}.', label=f'tab:{pipeline}_resultados_generales' )
    latex2 = metricasRendimiento[['Metaheuristica','Promedio_Error_Porcentual', 'Desviacion_Estandar_Porcentual','Intervalo_de_Confianza_Error_Absoluto_Minimo','Intervalo_de_Confianza_Error_Absoluto_Maximo', 'Promedio_Error_Abs','Desviacion_Estandar_Abs', 'Intervalo_de_Confianza_Error_Porcentual_Minimo', 'Intervalo_de_Confianza_Error_Porcentual_Maximo']].to_latex(index=False, column_format='lrrrrrrrrr', caption=f'Métricas de rendimiento {pipeline}.', label=f'tab:{pipeline}_metricas' )
    latex3 = desempeñoPorSolver[['Metaheuristica', 'TotalExperimentos', 'Tasa de Fallo']].sort_values(by='Tasa de Fallo').to_latex(index=False, column_format='lrrr', caption=f'Tasa de fallo del procesado {pipeline}.', label=f'tab:{pipeline}_fallos' )

    ## escribir tablas

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


## requieren un estado global o que le pase todas las pipelines de una en main. Habra que pensar que es mejor
def plotsGlobales(registroPipelines: dict):
    Plotter.graficos_globales(registroPipelines)

def actualizarDictPipelines(FallosTot: dict, registroPipelines, pipeline: str, dfProcesado: pd.DataFrame, resultadosAux: pd.DataFrame): 
    registroPipelines.append({
        "pipeline":     pipeline,
        "dfProcesado":  dfProcesado,
        "resultadosAux": resultadosAux,
        "FallosTot":    FallosTot,
    })

def _plots(totalExperimentos: int,pipeline: str,dfProcesado: pd.DataFrame,resultadosAux: pd.DataFrame) -> None:
    Plotter.graficos_por_pipeline(pipeline, dfProcesado, resultadosAux)


    


