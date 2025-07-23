def TS(solucionActual,mejorSolucion,mejorCosto,funcionVencindad, funcionEvaluacion,listaTabu,iteraciones, tamañoListaTabu):
    i = 0
    while i < iteraciones:
        "Se asume que la funcion de Vecindad retorna una lista de valores, de ser necesario, se puede implementar un loop para hacerla mas atomica"
        vecindad = funcionVencindad(solucionActual)
        for item in vecindad:
            if item in listaTabu:
                vecindad.remove(item)
        "idem que el comentario anterior"
        vecino, costoVecino = funcionEvaluacion(vecindad)

        if costoVecino < mejorCosto:
            mejorSolucion, mejorCosto = vecino, costoVecino
        if len(listaTabu) > tamañoListaTabu:
            listaTabu.pop(0)
        i = i + 1
    return solucionActual, mejorSolucion, mejorCosto