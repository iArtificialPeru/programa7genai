import functions_framework
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

@functions_framework.http
def gendata(request):
    request_args = request.args
    
    os.environ["OPENAI_API_KEY"] = "ingrese-aqui-su-api"
    
    rw = request_args['review']
    client = request_args['cliente']
    
    llm = ChatOpenAI(temperature=0.5)
    
    first_prompt = ChatPromptTemplate.from_template(
    """analiza el sentimiento de la siguiente reseña
    y devuelve el resultado en una sola palabra, positivo o negativo {review}""")

    second_prompt = ChatPromptTemplate.from_template(
    """Genera una respuesta de agradecimiento para {client} en formato de correo para la siguiente reseña positiva {review}""")

    third_prompt = ChatPromptTemplate.from_template(
    """Genera una respuesta de disculpas para {client} en formato de correo para la siguiente reseña negativa {review}""")

    #cadena 1 analisis de sentimiento
    chain_one = LLMChain(llm=llm, prompt=first_prompt,
                     output_key= "sentiment")
    result_one = chain_one.run({"review": rw})

    #cadena 2 condicional
    if "positivo" in result_one.lower():
        chain_two = LLMChain(llm=llm, prompt=second_prompt,
                     output_key= "response")
    elif "negativo" in result_one.lower():
        chain_two = LLMChain(llm=llm, prompt=third_prompt,
                     output_key= "response")
    result_chain = chain_two.run({"review": rw, "client": client})
    return result_chain
