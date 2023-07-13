from llama_index import SimpleDirectoryReader, GPTListIndex, \
    GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext, StorageContext, load_index_from_storage
from langchain import OpenAI
import gradio as gr
import openai
import sys
import os

# openai.api_key = os.getenv()
os.environ["OPENAI_API_KEY"] = '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'
openai.api_key = os.getenv("OPENAI_API_KEY")


def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    store_index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    store_index.storage_context.persist(persist_dir='./storage')

    return store_index


def chatbot(input_text):
    storage_context = StorageContext.from_defaults(persist_dir='./storage')
    storage_index = load_index_from_storage(storage_context)
    query_engine = storage_index.as_query_engine()
    response = query_engine.query(input_text)
    return response


iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Бот для Николая :)")

index = construct_index("docs")
chatbot("How do you say real estate in Russian?")
iface.launch(share=True)
