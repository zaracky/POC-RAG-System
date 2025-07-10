import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_mistralai import MistralAIEmbeddings
from tqdm import tqdm
from langchain.schema import Document

# Charger la clé API à partir du fichier .env
load_dotenv()
api_key = os.getenv('MISTRAL_AI_KEY')
embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

def nettoyer_texte(texte):
    if not texte or not isinstance(texte, str):
        return ""
    texte = BeautifulSoup(texte, "html.parser").get_text()
    texte = texte.lower()
    texte = re.sub(r'[^\w\s.,!?;:\'\"À-ÿ]', ' ', texte)
    return ' '.join(texte.split())

def obtenir_evenements_structures():
    start_year = 2025
    location = "Occitanie"
    results = []
    event_types = ["cinema", "festival", "concert", "danse", "spectacle", "théâtre", "jazz", "exposition",
                   "animation", "rock", "humour", "jeu", "ateliers", "peinture", "cirque", "chanson", "lecture",
                   "livre", "photographie", "film", "conte", "dessin", "chant", "art", "musique", "poésie"]

    for event_type in event_types:
        url = f"https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records?limit=1&refine=keywords_fr%3A%22{event_type}%22&refine=firstdate_begin%3A%22{start_year}%22&refine=location_region%3A%22{location}%22"
        response = requests.get(url)
        total_count = response.json().get("total_count", 0)

        for offset_index in range(int(total_count / 100) + 1):
            offset_url = f"https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records?limit=100&offset={offset_index * 100}&refine=keywords_fr%3A%22{event_type}%22&refine=firstdate_begin%3A%22{start_year}%22&refine=location_region%3A%22{location}%22"
            offset_response = requests.get(offset_url)
            results += offset_response.json().get("results", [])
            time.sleep(0.5)

    df = pd.DataFrame.from_dict(results)

    # Colonnes attendues, ajout avec valeurs par défaut si manquantes
    expected_cols = {
        "uid": "",
        "title_fr": "",
        "description_fr": "",
        "location_name": "",
        "location_address": "",
        "location_city": "",
        "location_postalcode": "",
        "daterange_fr": "",
        "keywords_fr": "",
        "firstdate_begin": pd.NaT,
        "lastdate_end": pd.NaT
    }
    for col, default in expected_cols.items():
        if col not in df.columns:
            df[col] = default

    # Nettoyage & filtrage
    df.drop_duplicates(subset="uid", inplace=True)
    df.dropna(subset=["uid", "title_fr", "description_fr"], inplace=True)

    # Conversion dates
    df["firstdate_begin"] = pd.to_datetime(df["firstdate_begin"], errors="coerce")
    df["lastdate_end"] = pd.to_datetime(df["lastdate_end"], errors="coerce")
    df["date_fin"] = df["lastdate_end"]  # clé pour filtrage index_faiss.py

    date_limit = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=365)
    df = df[df["firstdate_begin"] > date_limit]

    # Nettoyage texte
    df["description_fr"] = df["description_fr"].apply(nettoyer_texte)
    df["title_fr"] = df["title_fr"].apply(nettoyer_texte)

    # Transformation dates en string pour l'index
    df["firstdate_begin"] = df["firstdate_begin"].astype(str)
    df["lastdate_end"] = df["lastdate_end"].astype(str)

    # Construction du contenu complet
    df["content"] = (
        df["description_fr"] + " lieu: " + df["location_name"] +
        " adresse: " + df["location_address"] + " " + df["location_city"] + " " +
        df["location_postalcode"] + " dates: " + df["daterange_fr"] +
        " date de début: " + df["firstdate_begin"] + " date de fin: " + df["lastdate_end"] +
        " mots clés: " + df["keywords_fr"].astype(str)
    )

    return df



def generer_documents(df):
    documents = []
    for _, row in df.iterrows():
        content = (
            f"Titre: {row.get('title_fr', '')}\n"
            f"Description: {row.get('description_fr', '')}\n"
            f"Lieu: {row.get('location_name', '')} - {row.get('location_address', '')}, "
            f"{row.get('location_postalcode', '')} {row.get('location_city', '')}\n"
            f"Dates: {row.get('firstdate_begin', '')} - {row.get('lastdate_end', '')}\n"
            f"Mots-clés: {', '.join(row.get('keywords', [])) if isinstance(row.get('keywords'), list) else row.get('keywords', '')}"
        )
        
        metadata = {
            "source": "opendatasoft",
            "id": row.get("uid", ""),
            "title": row.get("title_fr", ""),
            "description": row.get("description_fr", ""),
            "firstdate_begin": row.get("firstdate_begin", ""),
            "lastdate_end": row.get("lastdate_end", ""),
            "date_fin": row.get("date_fin", ""),
            "location_name": row.get("location_name", ""),
            "location_address": row.get("location_address", ""),
            "location_district": row.get("location_district", ""),
            "location_postalcode": row.get("location_postalcode", ""),
            "location_city": row.get("location_city", ""),
            "location_description": row.get("location_description_fr", ""),
            "keywords": row.get("keywords", []),
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

    

def decouper_documents(documents):
    text_splitter = SemanticChunker(embeddings)
    splitted_docs = []
    for doc in tqdm(documents, desc="Découpe des documents", unit="document"):
        splitted_docs.extend(text_splitter.create_documents([doc.page_content]))
    return splitted_docs
