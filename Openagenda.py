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

def sanitize_text(text):
    """
    Nettoyer un texte pour éliminer les éléments indésirables avant de l'utiliser dans une base de données vectorielle.

    Parameters:
    text (str): Le texte à nettoyer.
    
    Returns:
    str: Le texte nettoyé.
    """
    # Supprimer le HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Convertir en minuscules
    text = text.lower()
    # Supprimer les caractères spéciaux et les emoji tout en gardant les accents
    text = re.sub(r'[^\w\s.,!?;:\'\"À-ÿ]', ' ', text)
    # Normaliser les espaces
    text = ' '.join(text.split())
    return text


def fetch_cultural_events_by_location(location):
    """
    Récupérer les événements culturels d'une région spécifique en utilisant l'API OpenAgenda.

    Parameters:
    location (str): La région pour laquelle récupérer les événements (par exemple, 'Occitanie').

    Returns:
    pd.DataFrame: Un DataFrame contenant les événements culturels nettoyés.
    """
    start_year = 2024
    results = []
    event_types = ["cinema", "festival", "concert", "danse", "spectacle", "théâtre", "jazz", "Exposition",
                   "animation", "rock", "humour", "jeu", "ateliers", "peinture", "cirque", "chanson", "lecture",
                   "livre", "photographie", "cinéma", "film", "conte", "dessin", "chant", "art", "musique", "poésie"]

    # Parcours des types d'événements et récupération des données depuis l'API
    for event_type in event_types:
        url = f"https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records?limit=1&refine=keywords_fr%3A%22{event_type}%22&refine=firstdate_begin%3A%22{str(start_year)}%22&refine=location_region%3A%22{location}%22" 
        response = requests.get(url)
        total_count = response.json()["total_count"]

        for offset_index in range(int(total_count / 100) + 1):
            offset_url = f"https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records?limit=100&offset={str(offset_index * 100)}&refine=keywords_fr%3A%22{event_type}%22&refine=firstdate_begin%3A%222024%22&refine=location_region%3A%22{location}%22"
            offset_response = requests.get(offset_url)
            offset_results = offset_response.json()["results"]
            results = results + offset_results
            time.sleep(0.5)

    # Transformer les données en DataFrame et nettoyage
    df = pd.DataFrame.from_dict(results)
    df.drop_duplicates(subset="uid", inplace=True)
    df.dropna(subset=["uid", "title_fr", "description_fr"], inplace=True)

    df["firstdate_begin"] = pd.to_datetime(df["firstdate_begin"])
    date_limit = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=365)
    df = df[df['firstdate_begin'] > date_limit]

    df["firstdate_begin"] = df["firstdate_begin"].astype(str)
    df["description_fr"] = df['description_fr'].apply(sanitize_text)
    df["content"] = df["description_fr"] + " lieu: " + df["location_name"] + " adresse: " + df["location_address"] + " " + df["location_city"] + " " + df["location_postalcode"] + " dates: " + df["daterange_fr"] + " date de début: " + df["firstdate_begin"] + " date de fin: " + df["lastdate_end"] + " mots clés: " + df["keywords_fr"].astype(str)

    return df


def fetch_example_events_for_index():
    """
    Récupérer un exemple d'événements pour l'indexation depuis l'API OpenAgenda.

    Returns:
    pd.DataFrame: Un DataFrame contenant un échantillon d'événements.
    """
    url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records?limit=100&refine=keywords_fr%3A%22Recrutement%22&refine=firstdate_begin%3A%222024%22&refine=location_region%3A%22Occitanie%22"
    response = requests.get(url)
    results = response.json()["results"]

    df = pd.DataFrame.from_dict(results)
    df.drop_duplicates(subset="uid", inplace=True)
    df.dropna(subset=["uid", "title_fr", "description_fr"], inplace=True)

    df["firstdate_begin"] = pd.to_datetime(df["firstdate_begin"])
    date_limit = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=365)
    df = df[df['firstdate_begin'] > date_limit]

    df["description_fr"] = df['description_fr'].apply(sanitize_text)
    df["content"] = df["description_fr"] + " \n lieu: " + df["location_name"] + " \n adresse: " + df["location_address"] + " " + df["location_city"] + " " + df["location_postalcode"] + " \ndates: " + df["daterange_fr"]

    return df


def convert_dataframe_to_documents(df):
    """Convertit un DataFrame en une liste de Documents Langchain.

    Parameters:
    df (DataFrame): Un DataFrame contenant les événements à convertir.

    Returns:
    List: Liste de documents Langchain.
    """
    documents = []
    for _, row in df.iterrows():
        content = row["content"]
        if pd.isna(content):
            content = f"description: {row['description_fr']} \nlieu: {row['location_name']} \nadresse: {row['location_address']} {row['location_city']} {row['location_postalcode']} \ndates: {row['daterange_fr']}"

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": "opendatasoft",
                    "id": row["uid"],
                    "title": sanitize_text(row["title_fr"]),
                    "description": row["description_fr"],
                    "firstdate_begin": row["firstdate_begin"],
                    "firstdate_end": row["firstdate_end"],
                    "lastdate_begin": row["lastdate_begin"],
                    "lastdate_end": row["lastdate_end"],
                    "location_coordinates": row["location_coordinates"],
                    "location_name": row["location_name"],
                    "location_address": row["location_address"],
                    "location_district": row["location_district"],
                    "location_postalcode": row["location_postalcode"],
                    "location_city": row["location_city"],
                    "location_description": row["location_description_fr"]
                }
            )
        )
    return documents


def split_documents_into_chunks(docs):
    """Découpe les documents en petits morceaux sémantiquement cohérents.

    Args:
        docs (List): Liste de documents Langchain.

    Returns:
        List: Liste de documents divisés.
    """
    text_splitter = SemanticChunker(MistralAIEmbeddings())
    splitted_docs = text_splitter.create_documents(docs)
    return splitted_docs
