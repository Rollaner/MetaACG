import random
import itertools
from string import Template
import pandas as pd
import json
## To-Do: Revisar templates. (Sanity check)
templateEval = Template("""TASK:GENERATE_COMPONENT_HEURISTIC_EVAL
PROBLEM_DEF:
---
$problema
---
SAMPLE_SOL:
$sampleSol
---
OUTPUT_FORMAT_STRICT:JSON.MUST_CONTAIN_2_KEYS.
OUTPUT_JSON_SCHEMA:{"REPRESENTATION": "...", "EVAL_CODE": "..."}

EVAL_CODE_DEF:PYTHON_FUNC.NAME=evaluate_solution.ARGS=1(solution).RET_NUM_FITNESS.SIG=def evaluate_solution(solution):

CRITICAL_INSTRUCTIONS:
1.CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED.
2.NO_ADD_TEXT_OR_EXPLANATION_OUTSIDE_THE_STRICT_OUTPUT_JSON.
3.EVAL_CODE_SIG_MUST_BE_def evaluate_solution(solution):
4.GENERATED_CODE_MUST_BE_COMPATIBLE_WITH_SAMPLE_SOL
5.CODE_KEYS_MUST_INCLUDE_IMPORTS.
6.NO_GLOBAL_VAR_REFERENCE_OR_DEFINITION_PROBLEM_DATA_MUST_BE_INTERNAL.
7.EXTERNAL_DATA_MUST_BE_PASSED_AS_ARGUMENT_DATA_MUST_BE_ENCODED_OR_EMBEDDED.
                        
INSPIRATIONS:
$Inspirations                        
""")

templatePerturb= Template("""TASK:GENERATE_COMPONENT_HEURISTIC_PERTURB                        
PROBLEM_DEF:
---
$problema
---
SAMPLE_SOL:
$sampleSol
---
OUTPUT_FORMAT_STRICT:JSON.MUST_CONTAIN_2_KEYS.
OUTPUT_JSON_SCHEMA:{"REPRESENTATION": "...", "PERTURB_CODE": "..."}

PERTURB_CODE_DEF:PYTHON_FUNC.NAME=perturb_solution.ARGS=1(solution).OP_PERTURBED_SOLUTION.SIG=def perturb_solution(solution):

CRITICAL_INSTRUCTIONS:
1.CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED.
2.NO_ADD_TEXT_OR_EXPLANATION_OUTSIDE_THE_STRICT_OUTPUT_JSON.
3.PERTURB_CODE_SIG_MUST_BE_def perturb_solution(solution):
4.GENERATED_CODE_MUST_BE_COMPATIBLE_WITH_SAMPLE_SOL
5.CODE_KEYS_MUST_INCLUDE_IMPORTS.
6.NO_GLOBAL_VAR_REFERENCE_OR_DEFINITION_PROBLEM_DATA_MUST_BE_INTERNAL.
7.EXTERNAL_DATA_MUST_BE_PASSED_AS_ARGUMENT_DATA_MUST_BE_ENCODED_OR_EMBEDDED.
                        
INSPIRATIONS:
$Inspirations
""")

templateNB= Template("""TASK:GENERATE_COMPONENT_HEURISTIC_NB
PROBLEM_DEF:
---
$problema
---
SAMPLE_SOL:
$sampleSol
---
OUTPUT_FORMAT_STRICT:JSON.MUST_CONTAIN_2_KEYS.
OUTPUT_JSON_SCHEMA:{"REPRESENTATION": "...", "NB_CODE": "..."}

NB_CODE_DEF:PYTHON_FUNC.NAME=generate_neighbour.ARGS=1(solution).OP_NEIGHBOR_SOLUTION.SIG=def generate_neighbour(solution) -> ("NB_Type", "Movement_Type"):

CRITICAL_INSTRUCTIONS:
1.CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED.
2.NO_ADD_TEXT_OR_EXPLANATION_OUTSIDE_THE_STRICT_OUTPUT_JSON.
3.NB_CODE_SIG_MUST_BE_def generate_neighbour(solution) -> ("NB_Type", "Movement_Type"):
4.GENERATED_CODE_MUST_BE_COMPATIBLE_WITH_SAMPLE_SOL
5.CODE_KEYS_MUST_INCLUDE_IMPORTS.
6.NO_GLOBAL_VAR_REFERENCE_OR_DEFINITION_PROBLEM_DATA_MUST_BE_INTERNAL.
7.EXTERNAL_DATA_MUST_BE_PASSED_AS_ARGUMENT_DATA_MUST_BE_ENCODED_OR_EMBEDDED.
                        
INSPIRATIONS:
$Inspirations
""")

templateEvalUpdate = Template("""TASK:GENERATE_COMPONENT_HEURISTIC_EVAL
PROBLEM_DEF:
---
$problema
---
SAMPLE_SOL:
$sampleSol
---
OUTPUT_FORMAT_STRICT:JSON.MUST_CONTAIN_2_KEYS.
OUTPUT_JSON_SCHEMA:{"REPRESENTATION": "...", "EVAL_CODE": "..."}

EVAL_CODE_DEF:PYTHON_FUNC.NAME=evaluate_solution.ARGS=1(solution).RET_NUM_FITNESS.SIG=def evaluate_solution(solution):

CRITICAL_INSTRUCTIONS:
1.CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED.
2.NO_ADD_TEXT_OR_EXPLANATION_OUTSIDE_THE_STRICT_OUTPUT_JSON.
3.EVAL_CODE_SIG_MUST_BE_def evaluate_solution(solution):
4.GENERATED_CODE_MUST_BE_COMPATIBLE_WITH_SAMPLE_SOL
5.CODE_KEYS_MUST_INCLUDE_IMPORTS.
6.NO_GLOBAL_VAR_REFERENCE_OR_DEFINITION_PROBLEM_DATA_MUST_BE_INTERNAL.
7.EXTERNAL_DATA_MUST_BE_PASSED_AS_ARGUMENT_DATA_MUST_BE_ENCODED_OR_EMBEDDED.
                        
INSPIRATIONS:
$Inspirations    
RESULTS_FORMAT "{current, currentScore, best, bestScore}"
$Resultados
FEEDBACK:
$Feedback
""")

templatePerturbUpdate = Template("""TASK:GENERATE_COMPONENT_HEURISTIC_PERTURB
PROBLEM_DEF:
---
$problema
---
SAMPLE_SOL:
$sampleSol
---
OUTPUT_FORMAT_STRICT:JSON.MUST_CONTAIN_2_KEYS.
OUTPUT_JSON_SCHEMA:{"REPRESENTATION": "...", "PERTURB_CODE": "..."}

PERTURB_CODE_DEF:PYTHON_FUNC.NAME=perturb_solution.ARGS=1(solution).OP_PERTURBED_SOLUTION.SIG=def perturb_solution(solution):

CRITICAL_INSTRUCTIONS:
1.CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED.
2.NO_ADD_TEXT_OR_EXPLANATION_OUTSIDE_THE_STRICT_OUTPUT_JSON.
3.PERTURB_CODE_SIG_MUST_BE_def perturb_solution(solution):
4.GENERATED_CODE_MUST_BE_COMPATIBLE_WITH_SAMPLE_SOL
5.CODE_KEYS_MUST_INCLUDE_IMPORTS.
6.NO_GLOBAL_VAR_REFERENCE_OR_DEFINITION_PROBLEM_DATA_MUST_BE_INTERNAL.
7.EXTERNAL_DATA_MUST_BE_PASSED_AS_ARGUMENT_DATA_MUST_BE_ENCODED_OR_EMBEDDED.
                        
INSPIRATIONS:
$Inspirations
RESULTS_FORMAT "{current, currentScore, best, bestScore}"
$Resultados
FEEDBACK:
$Feedback
""")

templateNBUpdate = Template("""TASK:GENERATE_COMPONENT_HEURISTIC_NB
PROBLEM_DEF:
---
$problema
---
SAMPLE_SOL:
$sampleSol
---
OUTPUT_FORMAT_STRICT:JSON.MUST_CONTAIN_2_KEYS.
OUTPUT_JSON_SCHEMA:{"REPRESENTATION": "...", "NB_CODE": "..."}

NB_CODE_DEF:PYTHON_FUNC.NAME=generate_neighbour.ARGS=1(solution).OP_NEIGHBOR_SOLUTION.SIG=def generate_neighbour(solution) -> ("NB_Type", "Movement_Type"):

CRITICAL_INSTRUCTIONS:
1.CODE_SYNTAX_CORRECT_FUNC_SELF_CONTAINED.
2.NO_ADD_TEXT_OR_EXPLANATION_OUTSIDE_THE_STRICT_OUTPUT_JSON.
3.NB_CODE_SIG_MUST_BE_def generate_neighbour(solution) -> ("NB_Type", "Movement_Type"):
4.GENERATED_CODE_MUST_BE_COMPATIBLE_WITH_SAMPLE_SOL
5.CODE_KEYS_MUST_INCLUDE_IMPORTS.
6.NO_GLOBAL_VAR_REFERENCE_OR_DEFINITION_PROBLEM_DATA_MUST_BE_INTERNAL.
7.EXTERNAL_DATA_MUST_BE_PASSED_AS_ARGUMENT_DATA_MUST_BE_ENCODED_OR_EMBEDDED.
                        
INSPIRATIONS:
$Inspirations
RESULTS_FORMAT "{current, currentScore, best, bestScore}"
$Resultados
FEEDBACK:
$Feedback
""")
#2.FORMAT_AS_KEY_VALUE_PAIRS.EX: "E_CODE_PERF:O(n) for each step. Consider incremental evaluation."
#3.PINPOINT_SPECIFIC_COMPONENT_FLAWS.EX: "NB_CODE_FAIL_LOCAL_OPT:Operator too simple, suggest 2-opt."
#4.SUGGEST_SPECIFIC_IMPROVEMENTS.EX: "R_STR_INADEQUATE:Binary string causing poor exploration. Recommend a permutation."
feedbackTemplate= Template("""TASK:GENERATE_FEEDBACK_FOR_COMPONENT_REFINEMENT 
FEEDBACK_INSTRUCTIONS:
0.NON_FUNCTIONAL_CODE_FORBIDDEN_LOCAL_SOLVER_ERRROS_MUST_BE_CORRECTED_FIRST
1.GENERATE_CRITICAL_FEEDBACK.FOCUS_ON_WEAKNESSES_AND_IMPROVEMENTS.AVOID_POSITIVE_REINFORCEMENT.
2.FORMAT_AS_KEY_VALUE_PAIRS.EX: "E_CODE_PARSE_ERROR: Solution is string, while E_CODE expects an array"
3.PINPOINT_SPECIFIC_COMPONENT_FLAWS_SUGGEST_SPECIFIC_IMPROVEMENTS_TO_FLAWS. MAXIMUM_4_ITEMS_ALLOWED_PRIORIZE_COMPATIBILITY_OF_COMPONENTS_WITH_LOCAL_SOLVER
4.KNOWN_RANDOM_SOLUTION_AND_VALUE_GIVEN_USE_PYTHON_TOOL_TO_EVALUATE_COMPONENTS_TO_ASSERT_CORRECTNESS
5.LOCAL_SOLVER_DESIGNED_FOR_EVALUATION_EXTRA_OUTPUTS_ARE_EXPECTED
6.CODE_MUST_BE_COMPATIBLE_WITH_SAMPLE_SOLUTION_AND_LOCAL_SOLVER
PROBLEM_DEF:
---
$problema
---
TARGET_HEURISTIC_GENERAL_SIGNATURE= "def Heuristic(currentSolution,best, best_score, generate_neighbour, evaluate_solution, perturb_solution, other_params)".                       
COMPONENTS:
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
HillClimbing
$resultadosHC
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

def generateEvalPrompt(problemaID, inspiraciones, samplesol):
    prompt = templateEval.safe_substitute(
        problema=inspiraciones,
        sampleSol = samplesol, 
        Inspirations = 'NA'
    ) 
    return prompt
#Updatea el prompt, assume una sola fila por version. 

def updateEvalPrompt(defProblema, componentes, resultados, feedback,samplesol):
    prompt = templateEvalUpdate.safe_substitute(
        problema = defProblema,
        sampleSol = samplesol,
        Inspirations = 'NA',
        Resultados=resultados,
        Feedback = feedback
    )
    return prompt

def generateNBPrompt(problemaID, inspiraciones,samplesol):
    prompt = templateNB.safe_substitute(
         problema=inspiraciones,
        sampleSol = samplesol, 
        Inspirations = 'NA'
    ) 
    return prompt
#Updatea el prompt, assume una sola fila por version. 

def updateNBPrompt(defProblema, componentes, resultados, feedback,samplesol):
    prompt = templateNBUpdate.safe_substitute(
        problema = defProblema,
        sampleSol = samplesol,
        Inspirations = 'NA',
        Resultados=resultados,
        Feedback = feedback
    )
    return prompt


def generatePerturbPrompt(problemaID, inspiraciones, samplesol):
    prompt = templatePerturb.safe_substitute(
        problema=inspiraciones,
        sampleSol = samplesol, 
        Inspirations = 'NA'
    ) 
    return prompt
#Updatea el prompt, assume una sola fila por version. 

def updatePerturbPrompt(defProblema, componentes, resultados, feedback, samplesol):
    prompt = templatePerturbUpdate.safe_substitute(
        problema = defProblema,
        sampleSol = samplesol,
        Inspirations = 'NA',
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
def generateFeedbackPrompt(defProblema, Eval, Nb, Perturb, SampleSol, resultadosSA, resultadosILS, resultadosTS, knownSol, knownObj):
    prompt = feedbackTemplate.safe_substitute(
        problema=defProblema, 
        Eval=Eval, 
        NB=Nb,
        perturb=Perturb, 
        SampleSol=SampleSol,
        resultadosSA=resultadosSA,
        resultadosILS=resultadosILS, 
        resultadosTS=resultadosTS, 
        knownSol=knownSol, 
        knownScore=knownObj
    ) 
    return prompt