import tiktoken
import streamlit as st

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
input_text = st.text_area("🔹 Prompt (input per il modello):", height=200)
output_text = st.text_area("🔸 Completamento atteso (output del modello):", height=200)

if st.button("Calcola costo"):
    input_tokens = count_tokens(input_text, model)
    output_tokens = count_tokens(output_text, model)
    cost = estimate_cost(input_tokens, output_tokens, model)

    st.markdown(f"**📊 Token Input:** {input_tokens}")
    st.markdown(f"**📈 Token Output:** {output_tokens}")
    st.markdown(f"**💵 Costo stimato:** ${cost} USD")
