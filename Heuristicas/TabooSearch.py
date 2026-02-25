def TS(solucionActual, mejorSolucion, mejorCosto, datosProblema,funcionVencindad, funcionPerturbacion, funcionEvaluacion,iteraciones, sizeListaTabu, duracionTabu):
    i = 0
    listaTabu = {}
    mejorCosto = funcionEvaluacion(solucionActual, datosProblema)
    while i < iteraciones:
        returnBruto = funcionVencindad(solucionActual, datosProblema)
        vecindad = []
        if isinstance(returnBruto, (list, tuple)):
            for item in returnBruto:
                if (isinstance(item, (list, tuple))
                        and len(item) == 2
                        and item[1] != ('none', -1)
                        and item[1] != 'none'):
                    vecindad.append((item[0], item[1]))
        if not vecindad:
            solucionActual = funcionPerturbacion(solucionActual, datosProblema)
            i += 1
            continue
        mejorVecinoIteracion = None
        costoMejorVecinoIteracion = float('inf')
        movimientoSeleccionado = None
        for vecino, movimiento in vecindad:
            costoVecino = funcionEvaluacion(vecino, datosProblema)
            esTabu = movimiento in listaTabu and listaTabu[movimiento] > i
            if esTabu and costoVecino < mejorCosto:
                esTabu = False
            if not esTabu and costoVecino < costoMejorVecinoIteracion:
                costoMejorVecinoIteracion = costoVecino
                mejorVecinoIteracion = vecino
                movimientoSeleccionado = movimiento
        if mejorVecinoIteracion is None:
            solucionActual = funcionPerturbacion(solucionActual, datosProblema)
            i += 1
            continue
        solucionActual = mejorVecinoIteracion
        if costoMejorVecinoIteracion < mejorCosto:
            mejorSolucion = solucionActual
            mejorCosto = costoMejorVecinoIteracion
        if movimientoSeleccionado is not None:
            listaTabu[movimientoSeleccionado] = i + duracionTabu
            if len(listaTabu) > sizeListaTabu:
                oldest = next(iter(listaTabu))
                del listaTabu[oldest]
        for m in [m for m, exp in listaTabu.items() if exp <= i]:
            del listaTabu[m]
        i += 1
    return (solucionActual,funcionEvaluacion(solucionActual, datosProblema),mejorSolucion,mejorCosto)
