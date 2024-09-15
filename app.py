import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Carica la chiave API di OpenAI dalle variabili d'ambiente
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verifica se la chiave API è stata trovata
if not openai_api_key:
    st.error("La chiave API di OpenAI non è stata trovata. Assicurati di averla impostata correttamente nei secrets di Streamlit Cloud.")
else:
    # Inizializza il modello LLM
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    llm_with_tools = llm.bind_tools([])  # Se hai strumenti, puoi passarli qui

    # Definizione del nodo assistant
    sys_msg = SystemMessage(content="Sei un assistente utile incaricato di eseguire operazioni aritmetiche su un insieme di input.")
    
    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    # Costruzione del grafo
    builder = StateGraph(MessagesState)

    # Aggiunta dei nodi
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode([]))  # Aggiungi gli strumenti se ne hai

    # Definizione degli archi
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    # Inizializzazione del checkpointer per la memoria
    memory = MemorySaver()

    # Compilazione del grafo con il checkpointer
    react_graph_memory = builder.compile(checkpointer=memory)

    # Configurazione di Streamlit per gestire lo stato
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
