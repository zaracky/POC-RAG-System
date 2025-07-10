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

now = datetime.now(timezone.utc)
print("Date actuelle:", now)

# — Étape 1 : Charger les événements
print(" Récupération des événements...")
df_events = obtenir_evenements_structures()
df_events["lastdate_end"] = pd.to_datetime(df_events["lastdate_end"], errors="coerce", utc=True)
df_events["firstdate_begin"] = pd.to_datetime(df_events["firstdate_begin"], errors="coerce", utc=True)

print(" Filtrage des événements...")
df_events = df_events[
    (df_events["lastdate_end"] >= now) &
    (df_events["description_fr"].notnull()) &
    (df_events["title_fr"].notnull())
]

# — Étape 2 : Convertir en documents Langchain
print(" Conversion en objets Documents...")
documents = generer_documents(df_events)


# — Étape 3 : Découpage sémantique
print("✂ Découpage sémantique des documents...")
docs_chunked = decouper_documents(documents)

# — Étape 4 : Initialiser les embeddings
print(" Génération des embeddings (Mistral)...")
embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

# — Étape 5 : Indexation dans FAISS
print(" Indexation FAISS...")
vectorstore = FAISS.from_documents(docs_chunked, embeddings)

# — Étape 6 : Sauvegarde
print(" Sauvegarde locale de l'index FAISS...")
vectorstore.save_local("faiss_index")

# — Étape 7 : Test
query = "événements de jazz à Toulouse"
print(f"🔍 Recherche sémantique : {query}")
docs_retrieved = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(docs_retrieved, 1):
    print(f"\n Résultat {i}:")
    print(doc.page_content)
    print(" Métadonnées:", doc.metadata)
