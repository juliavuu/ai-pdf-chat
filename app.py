import streamlit as st
import os
from dotenv import load_dotenv

# --- TEIL 1: SCHLÜSSEL LADEN ---

# Versuch 1: Aus Streamlit Cloud "Secrets" laden
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Versuch 2: Lokal vom Laptop laden (.env Datei)
load_dotenv()

# --- TEIL 2: PRÜFEN OB ER DA IST ---
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("Kein API-Key gefunden! Bitte in Streamlit Secrets eintragen.")
    st.stop()

# --- HIER GEHT DEIN CODE WEITER ---
# (Ab hier kommen deine Imports wie PyPDF2, LangChain usw.)


