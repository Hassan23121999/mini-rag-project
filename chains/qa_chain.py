from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

def build_qa_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini")
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa
