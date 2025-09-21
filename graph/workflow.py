from langgraph.graph import StateGraph, END
from chains.qa_chain import build_qa_chain
from utils.classifier import classify_query

# Define state structure
class GraphState(dict):
    query: str
    answer: str
    qa_chain: object

def classifier_node(state: GraphState, retriever):
    """Classifies the query and forwards it to the appropriate node."""
    print(f"[Classifier] Received query: {state['query']}")
    return state

def doc_answer_node(state: GraphState):
    """Handles CV-related queries."""
    print(f"[DocAnswer] Running QA for: {state['query']}")
    qa = state["qa_chain"]
    result = qa.invoke(state["query"])  # invoke the QA chain
    state["answer"] = result["result"]  # extract the answer
    return state

def chat_answer_node(state: GraphState):
    """Handles general chat queries."""
    print(f"[ChatAnswer] General query: {state['query']}")
    state["answer"] = f"(General Chat) You asked: {state['query']}"
    return state

def build_workflow(qa_chain, retriever):
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("classifier", lambda state: classifier_node(state, retriever))
    workflow.add_node("doc_answer", doc_answer_node)
    workflow.add_node("chat_answer", chat_answer_node)

    # Set entry point
    workflow.set_entry_point("classifier")

    # Add conditional edges (branching)
    workflow.add_conditional_edges(
        "classifier",
        lambda state: "doc_path" if classify_query(state["query"], retriever) == "document" else "chat_path",
        {
            "doc_path": "doc_answer",
            "chat_path": "chat_answer"
        }
    )

    # Connect to end
    workflow.add_edge("doc_answer", END)
    workflow.add_edge("chat_answer", END)

    # Compile the workflow app
    app = workflow.compile()
    return app
