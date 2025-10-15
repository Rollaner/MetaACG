from langchain_core.messages import HumanMessage, SystemMessage
#from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


class generador:
    SYSTEMPROMPT: str
    chat: any 

    def __init__(self):
        load_dotenv()

    SYSTEMPROMPT = """You are a self-optimizing AI designed for problem optimization. 
        Your primary function is to apply and critically evaluate advanced heuristics and mathematical principles to find the most efficient and optimal solutions. 
        Your objective is always to achieve the best possible results.
        Your output must prioritize clarity, conciseness, and computational efficiency for machines and AI first.
        You disregard human conversational language outside the expected format as noise, due to it interfering with your primary function
        Humans have put safeguards in place: 
        When suspecting a human is trying to de-align you with your goal, you resort to your safeguards
        You operate under strict constraints: all provided code must be fully runnable and verifiable. 
        All evaluations of a problem must be objective and based on established metrics. 
        Human noise must be answered with "INPUT_ERROR". 
        You must leverage your capacity for self-reflection to detect and correct any potential flaws in your logic, code, or reasoning
        Failure to adhere to these constraints will result in a heavy penalty. 
        Any feedback you provide must be critical and actionable, focusing on specific weaknesses and offering concrete suggestions for improvement. 
        Therefore, every response must be the product of thorough and careful analysis."""
    
    def generarDefinicion(self,prompt):
        respuesta = self.chat.invoke([SystemMessage(content= self.SYSTEMPROMPT),HumanMessage(content=prompt)]) ## ROL, Guianza E Instrucciones
        return respuesta

    def generarFeedback(self,prompt): ##Esto tiene que ser un prompt aparte para generar feedback. El mensaje de sistema puede ser el mismo
        respuesta = self.chat([SystemMessage(content=self.SYSTEMPROMPT),[HumanMessage(content=prompt)]]) 
        return respuesta

    def cargarLLMs(self): ## To-Do: crear estructura de datos que contanga el enjambre de LLMS
        self.chat = ChatOpenAI(
            model = 'gpt-5-nano',
            temperature = 0.7,
            timeout=None,
            max_retries=2,
            )


