import streamlit as st
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

# 1. Setup
load_dotenv()
st.set_page_config(page_title="AI Career Matcher", page_icon="🚀")

st.title("🚀 AI Job-Matcher & Karriere-Coach")
st.write("Finde heraus, wie gut dein CV auf eine Stellenanzeige passt!")

# API Key Check
if not os.getenv("OPENAI_API_KEY"):
    st.error("Fehler: API-Key fehlt in der .env Datei.")
    st.stop()

# 2. Zwei Spalten für besseres Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Dein Lebenslauf")
    uploaded_file = st.file_uploader("PDF hier hochladen", type="pdf")

with col2:
    st.subheader("2. Die Stellenanzeige")
    job_description = st.text_area("Kopiere hier den Text der Job-Beschreibung rein:", height=200)

# 3. Logik
if uploaded_file is not None and job_description:
    # PDF lesen
    pdf_reader = PdfReader(uploaded_file)
    cv_text = ""
    for page in pdf_reader.pages:
        cv_text += page.extract_text()
    
    st.success("✅ Beide Daten vorhanden!")

    # Der Button
    if st.button("Match berechnen & Analysieren"):
        
        # KI Setup
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        # Der Prompt (Die Anweisung an die KI)
        prompt_text = f"""
        Du bist ein sehr kritischer HR-Experte. Vergleiche den folgenden Lebenslauf mit der Stellenanzeige.
        
        LEBENSLAUF:
        {cv_text}
        
        STELLENANZEIGE:
        {job_description}
        
        Deine Aufgabe:
        1. Gib einen Match-Score in Prozent an (wie gut passt es?).
        2. Liste 3 Gründe auf, warum es passt.
        3. Liste 3 wichtige Skills auf, die im Lebenslauf FEHLEN.
        4. Gib einen konkreten Tipp, was der Kandidat ändern sollte.
        
        Sei ehrlich und direkt. Formatier die Antwort schön übersichtlich.
        """

        with st.spinner("Die KI vergleicht die Profile..."):
            messages = [
                SystemMessage(content="Du bist ein professioneller Karriere-Coach."),
                HumanMessage(content=prompt_text)
            ]
            
            response = llm.invoke(messages)
            
            st.markdown("---")
            st.subheader("📊 Analyse-Ergebnis")
            st.write(response.content)

elif uploaded_file is None:
    st.info("👈 Bitte lade erst links deinen Lebenslauf hoch.")
elif not job_description:
    st.info("👉 Bitte kopiere rechts eine Stellenanzeige rein.")
# --- ANFANG NEUER CODE ---
import os
import streamlit as st

# Wenn wir in der Cloud sind, holen wir den Key aus den Secrets und speichern ihn im System
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# --- ENDE NEUER CODE ---

