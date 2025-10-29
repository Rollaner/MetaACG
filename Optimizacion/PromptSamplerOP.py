import random
import itertools
from string import Template
## To-Do: Revisar templates. (Sanity check)
templateSeed= Template("""TASK:GENERATE_COMPONENTS_HEURISTIC_INIT.REP,NB,EVAL
PROBLEM_DEF:
---
$problema
---
OUTPUT_FORMAT_STRICT:
"R_STR","NB_CODE","E_CODE"
R_STR_DESC:STRING.SOL_ENCODE.EX:BIN_STR,PERM_LIST,TREE_STR.
NB_CODE_DEF:PYTHON_FUNC.NAME=generate_neighbour.ARGS=1(solution).OP_NEIGHBOR_SOLUTION.SIG=def generate_neighbour(solution):
E_CODE_DEF:PYTHON_FUNC.NAME=evaluate_solution.ARGS=1(solution).RET_NUM_FITNESS.SIG=def evaluate_solution(solution):
CRITICAL_INSTRUCTIONS:
1.CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED.
2.NO_ADD_TEXT_EXPLANATION_FORMAT_OUTSIDE_CSV.
3.SOL_ARG_TYPE_MATCH_R_STR.
INSPIRATIONS:
$inspirations
""")

templateUpdate= Template("""TASK:GENERATE_COMPONENTS_HEURISTIC.REP,NB,EVAL
PROBLEM_DEF:
---
$problema
---
OUTPUT_FORMAT_STRICT:
"R_STR","NB_CODE","E_CODE"
R_STR_DESC:STRING.SOL_ENCODE.EX:BIN_STR,PERM_LIST,TREE_STR.
NB_CODE_DEF:PYTHON_FUNC.NAME=generate_neighbour.ARGS=1(solution).OP_NEIGHBOR_SOLUTION.SIG=def generate_neighbour(solution):
E_CODE_DEF:PYTHON_FUNC.NAME=evaluate_solution.ARGS=1(solution).RET_NUM_FITNESS.SIG=def evaluate_solution(solution):
CRITICAL_INSTRUCTIONS:
1.CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED.
2.NO_ADD_TEXT_EXPLANATION_FORMAT_OUTSIDE_CSV.
3.SOL_ARG_TYPE_MATCH_R_STR.
INSPIRATIONS:
$inspirations                         
RESULTS
$resultados
FEEDBACK:
$feedback
""")

feedbackTemplate= Template("""TASK:GENERATE_FEEDBACK 
FEEDBACK_INSTRUCTIONS:
1.GENERATE_CRITICAL_FEEDBACK.FOCUS_ON_WEAKNESSES_AND_IMPROVEMENTS.AVOID_POSITIVE_REINFORCEMENT.
2.FORMAT_AS_KEY_VALUE_PAIRS.EX: "E_CODE_PERF:O(n) for each step. Consider incremental evaluation."
3.PINPOINT_SPECIFIC_COMPONENT_FLAWS.EX: "NB_CODE_FAIL_LOCAL_OPT:Operator too simple, suggest 2-opt."
4.SUGGEST_SPECIFIC_IMPROVEMENTS.EX: "R_STR_INADEQUATE:Binary string causing poor exploration. Recommend a permutation."
PROBLEM_DEF:
---
$problema
---
COMPONENTS:
$componente
RESULTS
$resultados
OUTPUT_FORMAT_STRICT:
"COMPONENT_VERSION", "FEEDBACK" """)

def sampleComponenteDB(componenteDB, problemaID,seed):
    problemSubset = componenteDB[componenteDB['ProblemaID'] == problemaID]
    representaciones = componenteDB.columns[1]
    representacionesPosibles = random.choice(list(problemSubset[representaciones].unique()))
    matches = problemSubset[problemSubset[representaciones] == representacionesPosibles]
    if len(matches) == 0:
        return matches  # or return None / empty DF / some sentinel
    tamaño = min(5, len(matches))
    return matches.sample(n=tamaño, random_state=seed)

def sampleFeedbackDB(feedbackDB, problemaID, componentSample): ## La logica es la misma de momento, se separan las funciones en caso de que se necesite modificar a futuro
    feedbackSubset = feedbackDB[feedbackDB['ProblemaID'] == problemaID]
    versionesComponente = componentSample.iloc[:, 4]
    feedback = feedbackSubset[feedbackSubset.iloc[:,1].isin(versionesComponente)]
    return feedback

def sampleResultadoDB(resultadoDB, problemaID, componentSample): ## To-Do revisar que la sintaxis este correcta
    resultadoSubset = resultadoDB[resultadoDB['ProblemaID'] == problemaID]
    versionesComponente = componentSample.iloc[:, 4]
    resultados = resultadoSubset[resultadoSubset.iloc[:,1].isin(versionesComponente)]
    return resultados

def sampleProblemaDB(problemaDB,seed):
    return problemaDB.sample(n=1, random_state=seed) 

def combinarComponentes(db):
    ComponenteVecindad = db.iloc[:,2].astype(str).unique().tolist()
    ComponenteEvaluacion = db.iloc[:,3].astype(str).unique().tolist()
    combinaciones = list(itertools.product(ComponenteVecindad,ComponenteEvaluacion))
    return combinaciones

def generarStrings(dataframe):
    if 'Text' in dataframe.columns:
        return  "\n".join(dataframe['Text'].astype(str).tolist())
    else: return f"{len(dataframe)} items found."

def generateSeedPrompt(problemaSample,componenteDB,seed):
    random.seed(seed)
    problemaID = problemaSample.iloc[0,0]
    componentSample = sampleComponenteDB(componenteDB, problemaID,seed)
    assert componentSample.iloc[:, 0].nunique() <= 1, "Multiples problemas diferentes seleccionados"
    assert componentSample.iloc[:, 1].nunique() <= 1, "Multiples representaciones differentes seleccionadas"
    combinaciones = combinarComponentes(componentSample) #revisar si esto es imprimible como string
    inspiraciones = ", ".join(f"{v} - {e}" for v, e in combinaciones)
    prompt = templateSeed.safe_substitute(problema=problemaSample.iloc[0,1], inspirations=inspiraciones) 
    return prompt

def updatePrompt(problemaSample, componenteDB, resultDB, feedbackDB, seed): ## podriamos optimizar po medio de almacenar un cache de las variables que se vuelven a calcular
    random.seed(seed)
    problemaID = problemaSample.iloc[0,0]
    componentSample = sampleComponenteDB(componenteDB, problemaID,seed)
    assert componentSample.iloc[:, 0].nunique() <= 1, "Multiples problemas diferentes seleccionados"
    assert componentSample.iloc[:, 1].nunique() <= 1, "Multiples representaciones differentes seleccionadas"
    resultSample = sampleResultadoDB(resultDB, problemaID, componentSample)
    feedbackSample = sampleFeedbackDB(feedbackDB, problemaID, componentSample)
    resultados = generarStrings(resultSample)
    feedback = generarStrings(feedbackSample) 
    combinaciones = combinarComponentes(componentSample)
    inspiraciones = ", ".join(f"{v} - {e}" for v, e in combinaciones)
    prompt = templateUpdate.safe_substitute(problema=problemaSample.iloc[0,1], inspirations=inspiraciones, resultados=resultados, feedback=feedback) 
    return prompt
## Feedback tiene que estar enfocado en un solo set de componentes a la vez. El ultimo que fue generado
def generateFeedbackPrompt(problemaSample, componente, resultados):
    prompt = feedbackTemplate.safe_substitute(problema=problemaSample.iloc[0,1], componente=componente, resultados=resultados) 
    return prompt