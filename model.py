#from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
#from langchain.llms import CTransformers
from langchain_community.llms import CTransformers

from langchain.chains import RetrievalQA
import chainlit as cl


DB_FAISS_PATH= "vectorstores/db_faiss"

custom_prompt_template = """use the following info to answer questions. simply say no if you don't know the answers.

Context:{context}
Question: {question}

Only return the helpful answer below and nothing else
Helpful answer: 
"""
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    """

    prompt = PromptTemplate(template = custom_prompt_template, input_variable = ['context','question'])
    
    return prompt

def load_llm():
    llm=CTransformers(
        model ="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type= "llama",
        max_new_tokens = 512,
        temperature = 0.5
    )

    return llm

def retrieval_qa_chain(llm,qa_prompt,db):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type ="stuff",
        retriever = db.as_retriever(search_kwargs ={'k':2}),
        return_source_documents = True,
        chain_type_kwargs ={'prompt': qa_prompt}
    )

    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings (model_name = 'sentence-transformers/all-MiniLM-L6-v2',
                                        model_kwargs= {'device':'cpu'})
    
    #db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    llm = load_llm()
    
    qa_prompt =set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

def final_result(query):
    qa_result= qa_bot()
    response = qa_result({'query': query})
    return response

##chainlit ##
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg =cl.Message(content = "Starting the bot.....")
    await msg.send()
    msg.content ="Hello, Welcome to the  Healthcare chatbot. What do you want to know about malaria and diabetes?"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )

    cb.answer_reached = True
    #res = await chain.acall(message, callbacks =[cb])
    res = await chain.acall({'query':message.content}, callbacks=[cb])
    answer =res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    
    else:
        answer += f"\nNo Sources Found"

    await cl.Message(content=answer).send()