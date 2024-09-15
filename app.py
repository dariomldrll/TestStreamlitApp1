import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

#caricamento variabili d'ambiente

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")



#Definizione degli strumenti (tools)

def multiply(a: int, b: int) -> int:
    """Moltiplica a per b."""
    return a * b

def add(a: int, b: int) -> int:
    """Somma a e b."""
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a per b."""
    return a / b

tools = [add, multiply, divide]

#inizializzazione del modello con gli strumenti 

llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
llm_with_tools = llm.bind_tools(tools)

#definizione del nodo assistant

sys_msg = SystemMessage(content="Sei un assistente utile incaricato di eseguire operazioni aritmetiche su un insieme di input.")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Costruzione del grafo
builder = StateGraph(MessagesState)

# Aggiunta dei nodi
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Definizione degli archi
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Inizializzazione del checkpointer per la memoria
memory = MemorySaver()

# Compilazione del grafo con il checkpointer
react_graph_memory = builder.compile(checkpointer=memory)

 #Configurazione di Streamlit per Gestire lo Stato
#Streamlit offre st.session_state per mantenere lo stato tra le interazioni dell'utente.

#conficgurazione interfaccia utente

st.title("Assistente Matematico con LangGraph")

# Inizializzazione dello stato della conversazione
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input dell'utente
user_input = st.text_input("Tu:", "")

# Configurazione del thread_id per la memoria
thread_id = "1"  # Puoi utilizzare un identificatore unico per ogni sessione
config = {"configurable": {"thread_id": thread_id}}

if st.button("Invia") and user_input:
    # Aggiungi il messaggio dell'utente
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Esegui il grafo con lo stato aggiornato
    final_state = react_graph_memory.invoke({"messages": st.session_state.messages}, config)
    
    # Aggiorna i messaggi con la risposta dell'assistente
    st.session_state.messages = final_state["messages"]
    
    # Visualizza la conversazione
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            st.write(f"**Tu:** {message.content}")
        elif isinstance(message, AIMessage):
            st.write(f"**Assistente:** {message.content}")
        elif message.type == "tool":
            st.write(f"**Strumento {message.name}:** {message.content}")

