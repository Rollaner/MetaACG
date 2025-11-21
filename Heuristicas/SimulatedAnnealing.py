import random
import math


def SA(solucionActual,mejor, mejorCosto, funcionVencindad, funcionEvaluacion, temp, minTemp, factorEnfriamiento):
    mejorCosto = funcionEvaluacion(solucionActual)
    while(temp > minTemp):
        vecino, _ = funcionVencindad(solucionActual)
        costoVecino = funcionEvaluacion(vecino)
        costoActual = funcionEvaluacion(solucionActual)
        delta = costoVecino - funcionEvaluacion(solucionActual)
        if costoVecino < mejorCosto:
            mejor = vecino
            mejorCosto = costoVecino
        if delta < 0 or random.random() < math.exp(-delta/temp):
            solucionActual = vecino
        temp *= factorEnfriamiento
    return solucionActual, costoActual, mejor, mejorCosto