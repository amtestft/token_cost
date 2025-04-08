import tiktoken
import streamlit as st
import pandas as pd
from collections import Counter
import html

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

def tokenize_text(text, model="gpt-4o"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    decoded_tokens = [enc.decode([t]) for t in tokens]
    return tokens, decoded_tokens

def render_tokenized_html(decoded_tokens):
    # Mostra i token con sfondo colorato alternato
    colors = ["#e0f7fa", "#fff9c4", "#fce4ec", "#f3e5f5", "#e8f5e9"]
    html_out = ""
    for i, token in enumerate(decoded_tokens):
        color = colors[i % len(colors)]
        safe_token = html.escape(token).replace(" ", "&nbsp;")
        html_out += f'<span style=\"background-color:{color};padding:2px;margin:1px;display:inline-block;\">{safe_token}</span>'
    return html_out

def display_token_stats(text, model="gpt-4o"):
    tokens, decoded = tokenize_text(text, model)
    token_ids = tokens
    num_tokens = len(tokens)
    num_chars = len(text)
    num_words = len(text.split())

    freqs = Counter(decoded)
    top_tokens = freqs.most_common(5)

    st.markdown(f"**🔢 Token:** {num_tokens} | **📝 Parole:** {num_words} | **🔡 Caratteri:** {num_chars}")
    st.markdown("**🧠 Token visualizzati:**")
    st.markdown(render_tokenized_html(decoded), unsafe_allow_html=True)

    st.markdown("**📊 Top 5 token più frequenti:**")
    for tok, freq in top_tokens:
        st.markdown(f"`{tok}` – {freq} volte")

# Streamlit UI
st.title("💰 Token & Costo Estimator per OpenAI API")

model = st.selectbox("Seleziona modello:", ["gpt-4o", "gpt-3.5-turbo", "gpt-4.5-preview", "gpt-4o-mini"])

# Placeholder per prompt di esempio SEO
with open('first_agent_prompt.txt', 'r') as file:
    example_prompt = file.read().replace('\n', '')
input_text = st.text_area("🔹 Prompt (input per il modello):", value=example_prompt, height=200)
output_text = st.text_area("🔸 Completamento atteso (output del modello):", height=200)

# Caricamento CSV opzionale
uploaded_file = st.file_uploader("📁 Carica un file CSV con dati SEO", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='utf-8')
    st.markdown("Anteprima del dataset:")
    st.dataframe(df.head())
    dataset_text = df.to_csv(index=False)
else:
    dataset_text = "\n".join(["col1,col2,col3,...,col10"] + ["val1,val2,val3,...,val10" for _ in range(1000)])

if st.button("Calcola costo"):
    # Agente 1 - raffinamento del prompt utente
    agent1_input = input_text
    with open('refined_prompt.txt', 'r') as file:
        agent1_output = file.read().replace('\n', '')
    agent1_cost = estimate_cost(count_tokens(agent1_input, model), count_tokens(agent1_output, model), model)

    # Agente 2 - analisi dei dati SEO
    agent2_input = agent1_output + "\n" + dataset_text
    with open('refined_prompt.txt', 'r') as file:
        agent2_cost = file.read().replace('\n', '')
    agent2_cost = estimate_cost(count_tokens(agent2_input, model), count_tokens(agent2_output, model), model)

    # Agente 3 - generazione Google Sheet (testo strutturato tipo slide deck)
    agent3_input = agent2_output + "generate a csv file which rows are 'slides' (slide, 1, slide 2, etc.) and columns are fields of each slide that we want to populate with information coming from the analysis that you provided"
    with open("slides.txt", 'r') as file:
        seo_slide_text = file.read().replace('\n', '')
    seo_slide_text = estimate_cost(count_tokens(agent3_input, model), count_tokens(agent3_output, model), model)

    total_cost = round(agent1_cost + agent2_cost + seo_slide_text, 6)

    st.subheader("📊 Costi stimati per ciascun agente")
    st.markdown(f"**Agente 1 (Prompt Refiner):** ${agent1_cost}")
    st.markdown(f"**Agente 2 (Analisi SEO):** ${agent2_cost}")
    st.markdown(f"**Agente 3 (Output strutturato):** ${agent3_cost}")
    st.markdown("---")
    st.subheader("🔍 Tokenizzazione del Prompt Utente (Agente 1)")
    display_token_stats(agent1_input, model)
    st.markdown("---")
    st.markdown(f"**💰 Costo totale stimato:** ${total_cost} USD")
