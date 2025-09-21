import gradio as gr
from utils.pdf_loader import load_and_split_pdf
from utils.vectorstore import create_vectorstore
from chains.qa_chain import build_qa_chain
from graph.workflow import build_workflow
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Preload PDF (CV)
docs = load_and_split_pdf("sample/CV_Hassan_Saleem.pdf")
vectorstore = create_vectorstore(docs)
qa_chain = build_qa_chain(vectorstore)

# Create the workflow
workflow = build_workflow(qa_chain, vectorstore.as_retriever())

def chat(query):
    state = {"query": query, "qa_chain": qa_chain}
    result = workflow.invoke(state)
    return result["answer"]

iface = gr.Interface(fn=chat, inputs="text", outputs="text", title="CV Expert Q&A")
iface.launch()
