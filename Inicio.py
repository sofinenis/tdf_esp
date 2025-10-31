import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# 🌻 Configuración general
st.set_page_config(
    page_title="🌻 Buscador Girasol TF-IDF 🌞",
    page_icon="🌻",
    layout="wide"
)

# 🌞 Estilo visual alegre
st.markdown("""
<style>
body {
    background-color: #fff8dc;
}
.main {
    background-color: #fffbea;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 10px rgba(255, 215, 0, 0.4);
}
h1, h2, h3 {
    color: #d4a017;
    text-align: center;
    font-family: 'Comic Sans MS', cursive;
}
.stTextInput>div>div>input, .stTextArea>div>textarea {
    background-color: #fff7b2;
    border: 2px solid #f4c430;
    border-radius: 10px;
    color: #4a3000;
    font-weight: bold;
}
.stButton>button {
    background-color: #ffd54f;
    color: #4a3000;
    border-radius: 12px;
    border: 2px solid #e1a72a;
    font-weight: bold;
    transition: all 0.3s;
}
.stButton>button:hover {
    background-color: #ffeb3b;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# 🌻 Título
st.title("🌻 Buscador TF-IDF — Gira con la Luz del Conocimiento 🌞")
st.markdown("""
Este pequeño buscador 🌼 utiliza **TF-IDF** y **similitud del coseno**  
para encontrar el documento que mejor responde a tu pregunta.  
Como un girasol 🌻 buscando el sol, tus palabras buscarán el sentido 🌞
""")

# Documentos de ejemplo
default_docs = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""

# Stemmer
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# 🌻 Layout
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📝 Documentos (uno por línea):", default_docs, height=150)
    question = st.text_input("❓ Escribe tu pregunta:", "¿Dónde juegan el perro y el gato?")

with col2:
    st.markdown("### 🌻 Preguntas sugeridas 🌞")
    sugeridas = [
        "¿Dónde juegan el perro y el gato?",
        "¿Qué hacen los niños en el parque?",
        "¿Cuándo cantan los pájaros?",
        "¿Dónde suena la música alta?",
        "¿Qué animal maúlla durante la noche?"
    ]
    for q in sugeridas:
        if st.button(q, use_container_width=True):
            st.session_state.question = q
            st.rerun()

# Actualizar si se seleccionó una sugerida
if 'question' in st.session_state:
    question = st.session_state.question

# 🌼 Botón principal
if st.button("🌞 Analizar con Luz TF-IDF"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, min_df=1)
        X = vectorizer.fit_transform(documents)
        
        # 📊 Matriz TF-IDF
        st.subheader("📊 Matriz TF-IDF — Huellas de Luz en las Palabras 🌻")
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"🌼 Doc {i+1}" for i in range(len(documents))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)
        
        # 🔍 Similitud
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]
        
        # 🎯 Resultado
        st.subheader("🎯 Resultado del Girasol Buscador")
        st.markdown(f"**Tu pregunta:** {question}")
        
        if best_score > 0.01:
            st.success(f"🌻 **Respuesta:** {best_doc}")
            st.info(f"📈 Similitud: {best_score:.3f}")
        else:
            st.warning(f"🌥️ **Respuesta con baja confianza:** {best_doc}")
            st.info(f"📉 Similitud: {best_score:.3f}")

# 🌼 Pie de página
st.markdown("---")
st.markdown("🌻 Desarrollado con alegría y rayos de sol ☀️ usando Streamlit + TF-IDF 🌼")
