import tiktoken
import streamlit as st
import pandas as pd

# Funzione per contare i token
@st.cache_data
def count_tokens(text, model="gpt-4o"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# Funzione per stimare il costo
def estimate_cost(input_tokens, output_tokens, model="gpt-4o"):
    pricing = {
        "gpt-4o": {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
        "gpt-3.5-turbo": {"input": 0.50 / 1e6, "output": 1.50 / 1e6},
        "gpt-4.5-preview": {"input": 75.00 / 1e6, "output": 150.00 / 1e6},
        "gpt-4o-mini": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
    }
    if model not in pricing:
        return None

    rate = pricing[model]
    cost = input_tokens * rate["input"] + output_tokens * rate["output"]
    return round(cost, 6)

# Streamlit UI
st.title("💰 Token & Costo Estimator per OpenAI API")

model = st.selectbox("Seleziona modello:", ["gpt-4o", "gpt-3.5-turbo", "gpt-4.5-preview", "gpt-4o-mini"])

# Placeholder per prompt di esempio SEO
example_prompt = "Analizza i dati SEO per identificare le pagine più performanti e suggerire miglioramenti."
input_text = st.text_area("🔹 Prompt (input per il modello):", value=example_prompt, height=200)
output_text = st.text_area("🔸 Completamento atteso (output del modello):", height=200)

# Caricamento CSV opzionale
uploaded_file = st.file_uploader("📁 Carica un file CSV con dati SEO", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown("Anteprima del dataset:")
    st.dataframe(df.head())
    dataset_text = df.to_csv(index=False)
else:
    dataset_text = "\n".join(["col1,col2,col3,...,col10"] + ["val1,val2,val3,...,val10" for _ in range(1000)])

if st.button("Calcola costo"):
    # Agente 1 - raffinamento del prompt utente
    agent1_input = input_text
    agent1_output = "Prompt raffinato basato sull'intento utente."
    agent1_cost = estimate_cost(count_tokens(agent1_input, model), count_tokens(agent1_output, model), model)

    # Agente 2 - analisi dei dati SEO
    agent2_input = agent1_output + "\n" + dataset_text
    agent2_output = "Analisi dettagliata dei dati SEO con spiegazioni e insight."
    agent2_cost = estimate_cost(count_tokens(agent2_input, model), count_tokens(agent2_output, model), model)

    # Agente 3 - generazione Google Sheet (testo strutturato tipo slide deck)
    agent3_input = agent2_output
    agent3_output = "Slide 1: panoramica SEO\nSlide 2: pagine performanti\nSlide 3: raccomandazioni"
    agent3_cost = estimate_cost(count_tokens(agent3_input, model), count_tokens(agent3_output, model), model)

    total_cost = round(agent1_cost + agent2_cost + agent3_cost, 6)

    st.subheader("📊 Costi stimati per ciascun agente")
    st.markdown(f"**Agente 1 (Prompt Refiner):** ${agent1_cost}")
    st.markdown(f"**Agente 2 (Analisi SEO):** ${agent2_cost}")
    st.markdown(f"**Agente 3 (Output strutturato):** ${agent3_cost}")
    st.markdown("---")
    st.markdown(f"**💰 Costo totale stimato:** ${total_cost} USD")
