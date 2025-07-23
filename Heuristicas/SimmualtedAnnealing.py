import random
import math


def SA(solucionActual,mejor, mejorCosto, funcionVencindad, funcionEvaluacion, temp, minTemp, factorEnfriamiento):
    while(temp > minTemp):
        vecino = funcionVencindad(solucionActual)
        costoVecino = funcionEvaluacion(vecino)
        delta = costoVecino - funcionEvaluacion(solucionActual)
        if delta < 0 or random.random() < math.exp(-delta/temp):
            solucionActual = vecino
        if costoVecino < mejorCosto:
            mejor, mejorCosto = vecino, costoVecino
        temp *= factorEnfriamiento
    return solucionActual, mejor, mejorCosto