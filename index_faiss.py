import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from Openagenda import (
    obtenir_evenements_structures,
    generer_documents,
    decouper_documents,
)

load_dotenv()
api_key = os.getenv('MISTRAL_AI_KEY')

# — Étape 1 : Charger les événements
print("📥 Récupération des événements...")
df_events = obtenir_evenements_structures()

# — Étape 2 : Convertir en documents Langchain
print("🧱 Conversion en objets Documents...")
documents = generer_documents(df_events)

# — Étape 3 : Découpage sémantique
print("✂️ Découpage sémantique des documents...")
docs_chunked = decouper_documents(documents)

# — Étape 4 : Initialiser les embeddings
print("🔎 Génération des embeddings...")
embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

# — Étape 5 : Indexation dans FAISS
print("📦 Indexation FAISS...")
vectorstore = FAISS.from_documents(docs_chunked, embeddings)

# — Étape 6 : Sauvegarde (optionnelle mais conseillée)
print("💾 Sauvegarde locale de l'index FAISS...")
vectorstore.save_local("faiss_index")

# — Étape 7 : Test de recherche
query = "événements de jazz à Toulouse"
print(f"🔍 Recherche sémantique : {query}")
docs_retrieved = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(docs_retrieved, 1):
    print(f"\n🔸 Résultat {i}:")
    print(doc.page_content)
    print("📍 Métadonnées:", doc.metadata)
