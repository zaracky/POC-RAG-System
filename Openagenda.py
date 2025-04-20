import re
import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_mistralai import MistralAIEmbeddings

# Charger la clé API à partir du fichier .env
load_dotenv()
api_key = os.getenv('MISTRAL_AI_KEY')

# Initialisation des embeddings Mistral AI
embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)


def nettoyer_texte(texte):
    """Nettoyer un texte (HTML, caractères spéciaux, normalisation)."""
    texte = BeautifulSoup(texte, "html.parser").get_text()
    texte = texte.lower()
    texte = re.sub(r'[^\w\s.,!?;:\'\"À-ÿ]', ' ', texte)
    texte = ' '.join(texte.split())
    return texte

def obtenir_evenements_structures():
    """Récupère les événements culturels filtrés et formatés depuis l'API OpenAgenda."""
    start_year = 2024
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
            offset_url = f"https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records?limit=100&offset={offset_index * 100}&refine=keywords_fr%3A%22{event_type}%22&refine=firstdate_begin%3A%222024%22&refine=location_region%3A%22{location}%22"
            offset_response = requests.get(offset_url)
            results += offset_response.json().get("results", [])
            time.sleep(0.5)

    df = pd.DataFrame.from_dict(results)
    df.drop_duplicates(subset="uid", inplace=True)
    df.dropna(subset=["uid", "title_fr", "description_fr"], inplace=True)

    df["firstdate_begin"] = pd.to_datetime(df["firstdate_begin"])
    date_limit = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=365)
    df = df[df["firstdate_begin"] > date_limit]

    df["firstdate_begin"] = df["firstdate_begin"].astype(str)
    df["description_fr"] = df["description_fr"].apply(nettoyer_texte)
    df["content"] = (
        df["description_fr"] + " lieu: " + df["location_name"] +
        " adresse: " + df["location_address"] + " " + df["location_city"] + " " +
        df["location_postalcode"] + " dates: " + df["daterange_fr"] +
        " date de début: " + df["firstdate_begin"] + " date de fin: " + df["lastdate_end"] +
        " mots clés: " + df["keywords_fr"].astype(str)
    )
    return df

def generer_documents(df):
    """Convertit un DataFrame en objets `Document` pour Langchain."""
    documents = []
    for _, row in df.iterrows():
        content = row["content"]
        if pd.isna(content):
            content = (
                f"description: {row['description_fr']} \nlieu: {row['location_name']} "
                f"\nadresse: {row['location_address']} {row['location_city']} {row['location_postalcode']} "
                f"\ndates: {row['daterange_fr']}"
            )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": "opendatasoft",
                    "id": row["uid"],
                    "title": nettoyer_texte(row["title_fr"]),
                    "description": row["description_fr"],
                    "firstdate_begin": row["firstdate_begin"],
                    "firstdate_end": row.get("firstdate_end", ""),
                    "lastdate_begin": row.get("lastdate_begin", ""),
                    "lastdate_end": row.get("lastdate_end", ""),
                    "location_coordinates": row.get("location_coordinates", ""),
                    "location_name": row["location_name"],
                    "location_address": row["location_address"],
                    "location_district": row.get("location_district", ""),
                    "location_postalcode": row["location_postalcode"],
                    "location_city": row["location_city"],
                    "location_description": row.get("location_description_fr", "")
                }
            )
        )
    return documents

def decouper_documents(documents):
    """
    Découpe les documents en morceaux plus petits, en affichant une barre de progression.

    Args:
        documents (List[Document]): Liste des documents à découper

    Returns:
        List[Document]: Liste des documents découpés
    """
    # Initialiser la barre de progression
    print("✂️ Découpage sémantique des documents...")
    text_splitter = SemanticChunker(embeddings)

    # Utilisation de tqdm pour la barre de progression
    splitted_docs = []
    for doc in tqdm(documents, desc="Découpe des documents", unit="document"):
        splitted_docs.extend(text_splitter.create_documents([doc.page_content]))  # Utilisation du texte brut

    return splitted_docs

