from collections import OrderedDict
import copy


def _contieneSentinel(obj, _visited=None): #Buscar de forma recursiva si es que tiene sentineles
    if _visited is None:
        _visited = set()
    obj_id = id(obj)
    if obj_id in _visited:
        return False
    _visited.add(obj_id)

    if obj is None:
        return True
    if isinstance(obj, str):
        return obj.strip().lower() == 'none'
    if isinstance(obj, bool):
        return False  
    if isinstance(obj, (int, float)):
        return obj == -1
    if isinstance(obj, (list, tuple, set, frozenset)):
        if len(obj) == 0:
            return True
        return any(_contieneSentinel(x, _visited) for x in obj)
    if isinstance(obj, dict):
        return any(
            _contieneSentinel(k, _visited) or _contieneSentinel(v, _visited)
            for k, v in obj.items()
        )
    return False


def esValido(item):
    if not isinstance(item, (list, tuple)):
        return False
    if len(item) != 2:
        return False
    _, movimiento = item
    return not _contieneSentinel(movimiento)

def normalizarMovimiento(movimiento):
    #to-do. Para eliminar error en linea 36. Hacemos un filtro y convertimos. 
    if movimiento is None:
        return None
    if isinstance(movimiento, (int, float, str, bool)): #Ya es valido, retornar como esta
        return movimiento
    if isinstance(movimiento, (list, tuple)):
        return tuple(normalizarMovimiento(x) for x in movimiento) #Normalizar todos elementos en lista/tupla
    if isinstance(movimiento, dict):
        return tuple(sorted((normalizarMovimiento(k), normalizarMovimiento(v)) for k, v in movimiento.items()),key=lambda par: str(par[0]))   #Normalizar todo en diccionario y ordenarlo
    if isinstance(movimiento, set):
        return frozenset(normalizarMovimiento(x) for x in movimiento) #Nomalizar tal y como esta
    try:
        hash(movimiento)
        return movimiento 
    except TypeError:     #Como fallback, se usa string para objectos exoticos
        return str(movimiento)

def TS(solucionActual, mejorSolucion, mejorCosto, datosProblema,funcionVencindad, funcionPerturbacion, funcionEvaluacion,iteraciones, sizeListaTabu, duracionTabu,maxPerturbConsecutivas = 5):
    ##Requiere una revision concienzuda. Puesto que todavia se cae muy seguido
    i = 0
    perturbConsecutivas = 0
    listaTabu = OrderedDict() #Para asegurar FIFO
    mejorCosto = funcionEvaluacion(solucionActual, datosProblema)
    while i < iteraciones:
        returnBruto = funcionVencindad(solucionActual, datosProblema)
        vecindad = []
        if isinstance(returnBruto, (list, tuple)):
            for item in returnBruto:
                if (esValido(item)):
                    vecindad.append((item[0], item[1]))
        if not vecindad:
            perturbConsecutivas += 1
            if perturbConsecutivas >= maxPerturbConsecutivas:
                # Diversificación: reiniciar desde la mejor solución conocida
                solucionActual = copy.deepcopy(mejorSolucion)
                perturbConsecutivas = 0
            else:
                solucionActual = funcionPerturbacion(solucionActual, datosProblema)
            i += 1
            continue
        mejorVecinoIteracion = None
        costoMejorVecinoIteracion = float('inf')
        movimientoSeleccionado = None
        for vecino, movimiento in vecindad:
            costoVecino = funcionEvaluacion(vecino, datosProblema)
            claveTabu = normalizarMovimiento(movimiento)
            esTabu = (claveTabu in listaTabu) and (listaTabu[claveTabu] > i)
            # Criterio de aspiración: aceptar tabú si mejora el mejor global
            if esTabu and costoVecino < mejorCosto:
                esTabu = False
            if not esTabu and costoVecino < costoMejorVecinoIteracion:
                costoMejorVecinoIteracion = costoVecino
                mejorVecinoIteracion = vecino
                movimientoSeleccionado = claveTabu

        if mejorVecinoIteracion is None:
            perturbConsecutivas += 1
            if perturbConsecutivas >= maxPerturbConsecutivas:
                # Diversificación: reiniciar desde la mejor solución conocida
                solucionActual = copy.deepcopy(mejorSolucion)
                perturbConsecutivas = 0
            else:
                solucionActual = funcionPerturbacion(solucionActual, datosProblema)
            i += 1
            continue
        perturbConsecutivas = 0
        solucionActual = mejorVecinoIteracion
        if costoMejorVecinoIteracion < mejorCosto:
            mejorSolucion = solucionActual
            mejorCosto = costoMejorVecinoIteracion

        if movimientoSeleccionado is not None:
            listaTabu[movimientoSeleccionado] = i + duracionTabu
            listaTabu.move_to_end(movimientoSeleccionado)
 
        expiradas = [movimientos for movimientos, duracionTabuActual in listaTabu.items() if duracionTabuActual <= i]
        for movimiento in expiradas:
            del listaTabu[movimiento]
 
        while len(listaTabu) > sizeListaTabu:
            listaTabu.popitem(last=False)
        i += 1
    return (solucionActual,funcionEvaluacion(solucionActual, datosProblema),mejorSolucion,mejorCosto)
