import streamlit as st
from google import genai
import pandas as pd
import numpy as np
import os
import re

try:
    import fasttext
except ImportError:
    # Alternative si fasttext-lite s'installe sous un autre nom
    import fasttext_lite as fasttext

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Assistant RGP Renault", page_icon="🚗", layout="wide")

# CSS pour cacher les menus Streamlit et épurer l'interface
hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# --- 2. RÉCUPÉRATION DES SECRETS (API KEY) ---
# Sur Streamlit Cloud, tu devras ajouter CLE_API dans les "Secrets" du dashboard
CLE_API = st.secrets.get("CLE_API") or os.environ.get("CLE_API")
client = genai.Client(api_key=CLE_API)
MODEL_ID = "gemini-3.1-flash-lite-preview"

CSV_PATH = 'Contracts_CY_newSCOPP - Database 15C.csv'
CC_LABEL_PATH = 'CC CC_Label.csv'
FASTTEXT_MODEL_PATH = 'model_cc_small.ftz'

# --- 3. CHARGEMENT DES RESSOURCES (CACHE) ---
@st.cache_resource
def load_all_resources():
    df, df_cc_labels, ft_model = None, None, None
    
    try:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH, encoding='latin1', sep=';')
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')
            
        if os.path.exists(CC_LABEL_PATH):
            try:
                df_cc_labels = pd.read_csv(CC_LABEL_PATH, sep=None, engine='python', encoding='utf-8')
            except:
                df_cc_labels = pd.read_csv(CC_LABEL_PATH, sep=None, engine='python', encoding='latin1')
            df_cc_labels.columns = df_cc_labels.columns.str.strip().str.replace('\ufeff', '')
            
        if os.path.exists(FASTTEXT_MODEL_PATH):
            ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
            
        return df, df_cc_labels, ft_model
    except Exception as e:
        st.error(f"Erreur de chargement des ressources : {e}")
        return None, None, None

df, df_cc_labels, ft_model = load_all_resources()

# --- 4. FONCTIONS MÉTIER (TES FONCTIONS) ---

def predict_with_small_model(text_input):
    if ft_model is None: return []
    text_clean = text_input.lower().replace("\n", " ").strip()
    labels, probabilities = ft_model.predict(text_clean, k=3)
    return [{"Code CC": l.replace("__label__", ""), "Confiance": f"{p*100:.2f}%"} for l, p in zip(labels, probabilities)]

def predict_buyer_ref(ref, Ntop=10):
    if df is None: return "Base de données non chargée.", None
    piece_data = df[df['Part_number'] == ref].copy()
    if piece_data.empty: return f"La référence {ref} est introuvable.", None
    
    piece_data['Contract_last_change'] = pd.to_datetime(piece_data['Contract_last_change'], dayfirst=True, errors='coerce')
    piece_data = piece_data.sort_values(by='Contract_last_change', ascending=False)
    
    dernier_row = piece_data.iloc[0]
    dernier_acheteur = dernier_row['Leader_buyer_name'] if pd.notna(dernier_row['Leader_buyer_name']) else dernier_row['CAP_creator_name']
    derniere_date = dernier_row['Contract_last_change'].strftime('%d/%m/%Y') if pd.notna(dernier_row['Contract_last_change']) else "Date inconnue"
    
    target_cc, target_pur_org = dernier_row['CC'], dernier_row['Purchasing_organization']
    df_espace = df[(df['CC'] == target_cc) & (df['Purchasing_organization'] == target_pur_org)].copy()
    df_espace['Contract_last_change'] = pd.to_datetime(df_espace['Contract_last_change'], dayfirst=True, errors='coerce')
    df_espace['Displayed_Buyer_Name'] = np.where(df_espace['Leader_buyer_name'].isna(), df_espace['CAP_creator_name'], df_espace['Leader_buyer_name'])
    
    top_buyers = df_espace.sort_values(by='Contract_last_change', ascending=False).drop_duplicates(subset=['Displayed_Buyer_Name']).head(Ntop).copy()
    
    top_buyers['CC'] = top_buyers['CC'].astype(str)
    df_cc_labels['CC'] = df_cc_labels['CC'].astype(str)
    result_final = top_buyers.merge(df_cc_labels[['CC', 'CC_Label']], on='CC', how='left')
    
    msg = f"### ANALYSE DE LA RÉFÉRENCE : {ref}\n1. Le dernier acheteur identifié est **{dernier_acheteur}** (le {derniere_date}).\n\n2. Top {Ntop} des acheteurs sur le segment **{target_cc}** :"
    return msg, result_final[['Contract_last_change', 'CC', 'CC_Label', 'Purchasing_organization', 'Displayed_Buyer_Name']]

def predict_buyer_cc(nom_piece, Ntop=10):
    predictions = predict_with_small_model(nom_piece)
    if not predictions: return f"⚠️ Impossible de prédire pour '{nom_piece}'.", None
    
    target_cc = str(predictions[0]['Code CC'])
    confiance = predictions[0]['Confiance']
    
    df_history = df.copy()
    df_history['CC'] = df_history['CC'].astype(str)
    df_espace = df_history[df_history['CC'] == target_cc].copy()
    
    if df_espace.empty: return f"❌ Aucun historique trouvé pour le code CC {target_cc}.", None
    
    df_espace['Contract_last_change'] = pd.to_datetime(df_espace['Contract_last_change'], dayfirst=True, errors='coerce')
    df_espace['Displayed_Buyer_Name'] = np.where(df_espace['Leader_buyer_name'].isna(), df_espace['CAP_creator_name'], df_espace['Leader_buyer_name'])
    
    top_buyers = df_espace.sort_values(by='Contract_last_change', ascending=False).drop_duplicates(subset=['Displayed_Buyer_Name']).head(Ntop).copy()
    df_cc_labels['CC'] = df_cc_labels['CC'].astype(str)
    result_final = top_buyers.merge(df_cc_labels[['CC', 'CC_Label']], on='CC', how='left')
    
    msg = f"### ANALYSE PAR NOM : '{nom_piece}'\n🎯 **Code CC prédit : {target_cc}** ({confiance})\n\nDerniers acheteurs actifs :"
    return msg, result_final[['Contract_last_change', 'CC', 'CC_Label', 'Purchasing_organization', 'Displayed_Buyer_Name']]

# --- 5. LOGIQUE DE RÉPONSE ---
def get_bot_response(user_input):
    prompt = f"Analyse : '{user_input}'. Formats: ACTION:REF | DATA:code ou ACTION:CC | DATA:nom ou ACTION:UNKNOWN."
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        decision = response.text.strip()
        
        if "ACTION:REF" in decision:
            ref = decision.split("DATA:")[1].strip()
            return predict_buyer_ref(ref)
        elif "ACTION:CC" in decision:
            nom = decision.split("DATA:")[1].strip()
            return predict_buyer_cc(nom)
        else:
            return "Précisez une référence (ex: 214G...) ou un nom de pièce.", None
    except Exception as e:
        return f"❌ Erreur : {str(e)}", None

# --- 6. INTERFACE UTILISATEUR (STREAMLIT) ---
st.title("🚗 Assistant Who is Buying What RGP")

# Initialisation historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "df" in message:
            st.dataframe(message["df"], use_container_width=True)

# Saisie utilisateur
if prompt := st.chat_input("Ex: Qui achète la 214G30212R ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyse en cours..."):
            res_text, res_df = get_bot_response(prompt)
            st.markdown(res_text)
            if res_df is not None:
                if 'Contract_last_change' in res_df.columns:
                    res_df['Contract_last_change'] = res_df['Contract_last_change'].dt.strftime('%d/%m/%Y')
                st.dataframe(res_df, use_container_width=True)
            
            # Sauvegarde dans l'historique
            st.session_state.messages.append({"role": "assistant", "content": res_text, "df": res_df})
