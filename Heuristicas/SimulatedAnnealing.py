import random
import math


def SA(solucionActual,mejor, mejorCosto,datosProblema, funcionVencindad, funcionEvaluacion, temp, minTemp, factorEnfriamiento):
    mejorCosto = funcionEvaluacion(solucionActual,datosProblema)
    while(temp > minTemp):
        vecino, _ = funcionVencindad(solucionActual,datosProblema)
        costoVecino = funcionEvaluacion(vecino,datosProblema)
        costoActual = funcionEvaluacion(solucionActual,datosProblema)
        delta = costoVecino - funcionEvaluacion(solucionActual,datosProblema)
        if costoVecino < mejorCosto:
            mejor = vecino
            mejorCosto = costoVecino
        if delta < 0 or random.random() < math.exp(-delta/temp):
            solucionActual = vecino
        temp *= factorEnfriamiento
    return solucionActual, costoActual, mejor, mejorCosto