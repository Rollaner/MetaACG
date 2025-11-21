import random
import itertools
from string import Template
import pandas as pd
import json
## To-Do: Revisar templates. (Sanity check)
templateSeed = Template("""TASK:GENERATE_COMPONENTS_HEURISTIC_INIT.REP,NB,PERTURB,EVAL
PROBLEM_DEF:
---
$problema
---
OUTPUT_FORMAT_STRICT:JSON.MUST_CONTAIN_5_KEYS.
OUTPUT_JSON_SCHEMA:{"REPRESENTATION": "...", "EVAL_CODE": "...", "NB_CODE": "...", "PERTURB_CODE": "...", "SAMPLE_SOL": "..."}
R_STR_DESC:STRING.SOL_ENCODE.EX:BIN_STR,PERM_LIST,TREE_STR.

TARGET_HEURISTIC_SA= "def SA(currentSolution,best, best_score, generate_neighbour, evaluate_solution, temp, minTemp, cooling_factor)".
HEURISTICS_VALUE_BEST_AS_LESSER_COST_USE_NEGATIVES_FOR_MAXIMIZATION_PROBLEMS

NB_CODE_DEF:PYTHON_FUNC.NAME=generate_neighbour.ARGS=1(solution).OP_NEIGHBOR_SOLUTION.SIG=def generate_neighbour(solution) -> ("NB_Type", "Movement_Type"):
PERTURB_CODE_DEF:PYTHON_FUNC.NAME=perturb_solution.ARGS=1(solution).OP_PERTURBED_SOLUTION.SIG=def perturb_solution(solution):
EVAL_CODE_DEF:PYTHON_FUNC.NAME=evaluate_solution.ARGS=1(solution).RET_NUM_FITNESS.SIG=def evaluate_solution(solution):

CRITICAL_INSTRUCTIONS:
1.CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED.
2.NO_ADD_TEXT_OR_EXPLANATION_OUTSIDE_THE_STRICT_OUTPUT_JSON.
3.SOL_ARG_TYPE_MATCH_R_STR.
4.NB_CODE_SIG_MUST_BE_def generate_neighbour(solution) -> ("NB_Type", "Movement_Type"):
5.PERTURB_CODE_SIG_MUST_BE_def perturb_solution(solution):
6.EVAL_CODE_SIG_MUST_BE_def evaluate_solution(solution):
7.SAMPLE_SOL_MATCH_R_STR.
8.CODE_KEYS_MUST_INCLUDE_IMPORTS.
9.NO_GLOBAL_VAR_REF_OR_DEF_PROBLEM_DATA_MUST_BE_INTERNAL.
10.EXTERNAL_DATA_MUST_BE_PASSED_AS_ARG_DATA_MUST_BE_ENCODED_OR_EMBEDDED.
                        
INSPIRATIONS:
"Objective function":
$Obj
"Evaluation Function":
$Eval
""")

templateSeedRaw = Template("""TASK:GENERATE_COMPONENTS_HEURISTIC_INIT.REP,NB,PERTURB,EVAL
PROBLEM_DEF:
---
$problema
---
OUTPUT_FORMAT_STRICT:JSON.MUST_CONTAIN_5_KEYS.
OUTPUT_JSON_SCHEMA:{"REPRESENTATION": "...", "EVAL_CODE": "...", "NB_CODE": "...", "PERTURB_CODE": "...", "SAMPLE_SOL": "..."}
R_STR_DESC:STRING.SOL_ENCODE.EX:BIN_STR,PERM_LIST,TREE_STR.

TARGET_HEURISTIC_SA= "def SA(currentSolution,best, best_score, generate_neighbour, evaluate_solution, temp, minTemp, cooling_factor)".
HEURISTICS_VALUE_BEST_AS_LESSER_COST_USE_NEGATIVES_FOR_MAXIMIZATION_PROBLEMS

NB_CODE_DEF:PYTHON_FUNC.NAME=generate_neighbour.ARGS=1(solution).OP_NEIGHBOR_SOLUTION.SIG=def generate_neighbour(solution) -> ("NB_Type", "Movement_Type"):
PERTURB_CODE_DEF:PYTHON_FUNC.NAME=perturb_solution.ARGS=1(solution).OP_PERTURBED_SOLUTION.SIG=def perturb_solution(solution):
EVAL_CODE_DEF:PYTHON_FUNC.NAME=evaluate_solution.ARGS=1(solution).RET_NUM_FITNESS.SIG=def evaluate_solution(solution):

CRITICAL_INSTRUCTIONS:
1.CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED.
2.NO_ADD_TEXT_OR_EXPLANATION_OUTSIDE_THE_STRICT_OUTPUT_JSON.
3.SOL_ARG_TYPE_MATCH_R_STR.
4.NB_CODE_SIG_MUST_BE_def generate_neighbour(solution) -> ("NB_Type", "Movement_Type"):
5.PERTURB_CODE_SIG_MUST_BE_def perturb_solution(solution):
6.EVAL_CODE_SIG_MUST_BE_def evaluate_solution(solution):
7.SAMPLE_SOL_MATCH_R_STR.
8.CODE_KEYS_MUST_INCLUDE_IMPORTS.
9.NO_GLOBAL_VAR_REF_OR_DEF_PROBLEM_DATA_MUST_BE_INTERNAL.
10.EXTERNAL_DATA_MUST_BE_PASSED_AS_ARG_DATA_MUST_BE_ENCODED_OR_EMBEDDED.
                    
""")

templateUpdate= Template("""TASK:GENERATE_COMPONENTS_HEURISTIC_INIT.REP,NB,PERTURB,EVAL
PROBLEM_DEF:
---
$problema
---
OUTPUT_FORMAT_STRICT:JSON.MUST_CONTAIN_5_KEYS.
OUTPUT_JSON_SCHEMA:{"REPRESENTATION": "...", "EVAL_CODE": "...", "NB_CODE": "...", "PERTURB_CODE": "...",, "SAMPLE_SOL": "..."}
R_STR_DESC:STRING.SOL_ENCODE.EX:BIN_STR,PERM_LIST,TREE_STR.

TARGET_HEURISTIC_SA= "def SA(currentSolution,best, best_score, generate_neighbour, evaluate_solution, temp, minTemp,cooling_factor)".
HEURISTICS_VALUE_BEST_AS_LESSER_COST_USE_NEGATIVES_FOR_MAXIMIZATION_PROBLEMS

NB_CODE_DEF:PYTHON_FUNC.NAME=generate_neighbour.ARGS=1(solution).OP_NEIGHBOR_SOLUTION.SIG=def generate_neighbour(solution) -> ("NB_Type", "Movement_Type"):
PERTURB_CODE_DEF:PYTHON_FUNC.NAME=perturb_solution.ARGS=1(solution).OP_PERTURBED_SOLUTION.SIG=def perturb_solution(solution):
EVAL_CODE_DEF:PYTHON_FUNC.NAME=evaluate_solution.ARGS=1(solution).RET_NUM_FITNESS.SIG=def evaluate_solution(solution):

CRITICAL_INSTRUCTIONS:
1.CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED.
2.NO_ADD_TEXT_OR_EXPLANATION_OUTSIDE_THE_STRICT_OUTPUT_JSON.
3.SOL_ARG_TYPE_MATCH_R_STR.
4.NB_CODE_SIG_MUST_BE_def generate_neighbour(solution) -> ("NB_Type", "Movement_Type"):
5.PERTURB_CODE_SIG_MUST_BE_def perturb_solution(solution):
6.EVAL_CODE_SIG_MUST_BE_def evaluate_solution(solution):
7.SAMPLE_SOL_MATCH_R_STR
8.CODE_KEYS_MUST_INCLUDE_IMPORTS.
9.NO_GLOBAL_VAR_REF_OR_DEF_PROBLEM_DATA_MUST_BE_INTERNAL.
10.EXTERNAL_DATA_MUST_BE_PASSED_AS_ARG_DATA_MUST_BE_ENCODED_OR_EMBEDDED.
                         
INSPIRATIONS:
"Representation":
$Rep
"Evaluation Function":
$Eval
"Neigbour Function":
$NB    
"Perturbation Function:"
$Perturb
"Sample Solution":
$SampleSol   
RESULTS_FORMAT "{current, currentScore, best, bestScore}"
$Resultados
FEEDBACK:
$Feedback
""")

feedbackTemplate= Template("""TASK:GENERATE_FEEDBACK 
FEEDBACK_INSTRUCTIONS:
0.FIX_LOCAL_SOLVER_ERRORS_FIRST
1.GENERATE_CRITICAL_FEEDBACK.FOCUS_ON_WEAKNESSES_AND_IMPROVEMENTS.AVOID_POSITIVE_REINFORCEMENT.
2.FORMAT_AS_KEY_VALUE_PAIRS.EX: "E_CODE_PERF:O(n) for each step. Consider incremental evaluation."
3.PINPOINT_SPECIFIC_COMPONENT_FLAWS.EX: "NB_CODE_FAIL_LOCAL_OPT:Operator too simple, suggest 2-opt."
4.SUGGEST_SPECIFIC_IMPROVEMENTS.EX: "R_STR_INADEQUATE:Binary string causing poor exploration. Recommend a permutation."
5.KNOWN_BEST_SOLUTION_AND_VALUE_GIVEN_USE_PYTHON_TOOL_TO_EVALUATE_GIVEN_EVAL_WITH_KNOWN_BEST_TO_ASSERT_CORRECTNESS
6.LOCAL_SOLVER_DESIGNED_FOR_EVALUATION_EXTRA_OUTPUTS_ARE_EXPECTED
7.DO_NOT_MENTION_KNOWN_BEST_SOLUTION_NOR_VALUE_IN_FEEDBACK_TO_PREVENT_CHEATING
PROBLEM_DEF:
---
$problema
---
TARGET_HEURISTIC_GENERAL_SIGNATURE= "def Heuristic(currentSolution,best, best_score, generate_neighbour, evaluate_solution, perturb_solution, other_params)".
HEURISTICS_VALUE_BEST_AS_LESSER_COST_USE_NEGATIVES_FOR_MAXIMIZATION_PROBLEMS                           
COMPONENTS:
"Representation":
$Rep
"Evaluation Function":
$Eval
"Neigbour Function":
$NB    
"Perturbation Function:"
$Perturb
"Sample Solution":
$SampleSol
RESULTS_FROM_LOCAL_SOLVER
Simulated_Annealing
$resultadosSA
Iterated_Local_Search
$resultadosILS
Taboo_Search
$resultadosTS
KNOWN_SOLUTION
$knownSol
EXPECTED_SCORE_FROM_KNOWN_SOLUTION
$knownScore
OUTPUT_FORMAT_STRICT:
"COMPONENT_VERSION", "FEEDBACK" """)


def sampleProblemaDB(problemaDB,seed):
    return problemaDB.sample(n=1, random_state=seed) 

def generarStrings(dataframe):
    if 'Text' in dataframe.columns:
        return  "\n".join(dataframe['Text'].astype(str).tolist())
    else: return f"{len(dataframe)} items found."

def generateRawSeedPrompt(problemaSample:pd.DataFrame):
    if isinstance(problemaSample, pd.DataFrame):
        problemaID = problemaSample.Instancia.iloc[0]
        inspiraciones=problemaSample.Respuesta.iloc[0]
        knownSol = problemaSample['Resultado esperado'].iloc[0]
        knownObj = problemaSample['Valor Objetivo'].iloc[0]
    else:
        problemaID = problemaSample.Instancia
        inspiraciones=problemaSample.Respuesta
        knownSol = problemaSample[7] #Resultado_esperado
        knownObj = problemaSample[8] #Valor_Objetivo
    prompt = templateSeedRaw.safe_substitute(
        problema=inspiraciones
    ) 
    return problemaID, inspiraciones, knownSol, knownObj, prompt

def generateSeedPrompt(problemaSample:pd.DataFrame):
    if isinstance(problemaSample, pd.DataFrame):
        problemaID = problemaSample.Instancia.iloc[0]
        inspiraciones=json.loads(problemaSample.Respuesta.iloc[0])
        knownSol = problemaSample['Resultado esperado'].iloc[0]
        knownObj = problemaSample['Valor Objetivo'].iloc[0]
    else:
        problemaID = problemaSample.Instancia
        inspiraciones=json.loads(problemaSample.Respuesta)
        knownSol = problemaSample[7] #Resultado_esperado
        knownObj = problemaSample[8] #Valor_Objetivo
    prompt = templateSeed.safe_substitute(
        problema=inspiraciones['MATH_DEF'], 
        Sol=inspiraciones['SOL_TYPE'],
        Obj=inspiraciones['OBJ_CODE'], 
        Eval = inspiraciones['EVAL_CODE']
    ) 
    return problemaID, inspiraciones['MATH_DEF'], knownSol, knownObj, prompt
#Updatea el prompt, assume una sola fila por version. 

def updatePromptOS(defProblema, componentes, resultados, feedback):
    prompt = templateUpdate.safe_substitute(
        problema = defProblema,
        Rep=componentes['REPRESENTATION'],
        Eval=componentes['EVAL_CODE'], 
        NB=componentes['NB_CODE'],
        Perturb=componentes['PERTURB_CODE'], 
        Resultados=resultados,
        Feedback = feedback
    )
    return prompt
# Este requiere que existan las DB de antemano. 
#(Para hacer mix and match se tiene que mezclar antes de que se haga la evaluacion y feedback, con la version siendo dependentiente de la iteracion)
#En consecuencia las versiones son enteros en el rango [0:Iteraciones]
def retornarCoincidentes(df:pd.DataFrame, problemaID, version):
    filaCoincidente = df[
        (df['ID_Problema'] == problemaID) &
        (df['Version'] == version)
    ]
    if filaCoincidente.empty:
        raise ValueError(f"No component set found for Problema ID: {problemaID} and Version: {version}")
    return filaCoincidente.iloc[0]

#Funcion para hacer mix and match. Debe llamarse antes de hacer una evaluacion. Se genera feedback con los componentes mezclados y los resultados de la evalucion
def sampleComponenteDB(componenteDB, problemaID,version,seed):
    subsetProblema = componenteDB[componenteDB['ID_Problema'] == problemaID]
    if subsetProblema.empty:
        return pd.DataFrame()
    random.seed(seed)
    try:
        representaciones = subsetProblema['Representacion'].unique()
        representacionEscogida = random.choice(representaciones)
    except IndexError:
        return pd.DataFrame()
    subsetCompatible = subsetProblema[subsetProblema['Representacion'] == representacionEscogida]
    try:
        eval = random.choice(subsetCompatible['Evaluacion'].unique())
        vecindad = random.choice(subsetCompatible['Vecindad'].unique())
        perturbacion = random.choice(subsetCompatible['Perturbacion'].unique())
    except IndexError:
        return pd.DataFrame()
    datosComponentes = {
        'ID_Problema': problemaID,
        'Representacion': representacionEscogida,
        'Evaluacion': eval,
        'Vecindad': vecindad,
        'Perturbacion': perturbacion,
        'Version': version 
    }
    return pd.DataFrame([datosComponentes])

## Feedback tiene que estar enfocado en un solo set de componentes a la vez. El ultimo que fue generado
def generateFeedbackPrompt(defProblema, componentes, resultadosSA, resultadosILS, resultadosTS, knownSol, knownObj):
    prompt = feedbackTemplate.safe_substitute(
        problema=defProblema, 
        Rep=componentes['REPRESENTATION'],
        Eval=componentes['EVAL_CODE'], 
        NB=componentes['NB_CODE'],
        perturb=componentes['PERTURB_CODE'], 
        SampleSol=componentes['SAMPLE_SOL'], 
        resultadosSA=resultadosSA,
        resultadosILS=resultadosILS, 
        resultadosTS=resultadosTS, 
        knownSol=knownSol, 
        knownScore=knownObj
    ) 
    return prompt