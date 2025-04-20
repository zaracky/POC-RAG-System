# POC-RAG-System: Chatbot Culturel Occitanie

Ce projet implémente un **chatbot culturel** capable de recommander des événements en Occitanie en se basant sur des données provenant d'Open Agenda, et d'interagir avec l'utilisateur de manière naturelle. Le chatbot utilise **FAISS** pour l'indexation des événements et **MistralAI** pour les embeddings et la génération de réponses via un modèle de langage.

## Fonctionnalités

- Recherche d'événements en Occitanie selon des critères spécifiques (type d'événement, lieu, date).
- Recommandations personnalisées en fonction de la question de l'utilisateur.
- Génération de réponses naturelles basées sur des données extraites de la base vectorielle.
- Tests unitaires intégrés via GitHub Actions pour assurer la qualité du code.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les prérequis suivants :

- Python 3.8+ (Testé avec Python 3.8, 3.9, 3.10)
- GitHub Actions pour l'exécution des tests unitaires
- Clé API pour **MistralAI** (inscription nécessaire sur [Mistral AI](https://mistral.ai))

## Installation

1. Clonez ce dépôt sur votre machine locale :

   ```bash
   git clone https://github.com/zaracky/POC-RAG-System.git
   cd chatbot-culturel-occitanie


2. Créez un environnement virtuel :

   ```bash
   python -m venv env
   source env/bin/activate  # sur Windows : env\Scripts\activate

3. Installez les dépendances nécessaires :

   ```bash
   pip install -r requirements.txt

4. Configurez votre clé API Mistral dans les variables d'environnement :

   ```bash
   export MISTRAL_API_KEY="votre-clé-api"

ou créez un fichier .env à la racine du projet avec la clé :

   - MISTRAL_API_KEY="votre-clé-api"


5. Exécutez le chatbot :

   ```bash
   python chatbot.py

Vous pouvez commencer à poser des questions au chatbot, par exemple :

   - Quels événements musicaux à Toulouse cette semaine ?

## Tests

Les tests unitaires sont intégrés via GitHub Actions et sont exécutés automatiquement lors de chaque push ou pull request. Pour les exécuter localement, utilisez :

   - python -m unittest discover tests/




## Structure du Projet

      ├── chatbot-culturel-occitanie/
         ├── chatbot.py                   
         ├── requirements.txt           
         ├── tests/         
            └── test.yml           
            └── test_chatbot.py
         └── Openagenda.py
         └── Index_faiss.py
         ├── index_faiss/  # Dossier contenant les index vectoriels

## Explications des fichiers et répertoires :
- chatbot.py : Contient le code principal pour faire fonctionner le chatbot et interagir avec l'utilisateur.

- index_faiss.py : Contient le code pour indexer les événements dans la base FAISS.

- openagenda.py : Gère l'accès et le nettoyage des données provenant d'OpenAgenda avant de les indexer.

- requirements.txt : Liste des dépendances nécessaires pour faire fonctionner le projet.

- index/ : Contient les fichiers d'index FAISS. Ce répertoire est utilisé pour stocker les index qui permettent d'effectuer des recherches rapides basées sur la similarité sémantique.

- tests/ : Contient les tests unitaires pour vérifier le bon fonctionnement du chatbot.


## GitHub Actions - CI/CD

Un pipeline GitHub Actions est configuré pour exécuter les tests unitaires lors de chaque push ou pull request. Il utilise le fichier test.yml dans le répertoire .github/workflows/ pour automatiser cette tâche.

## Contributions

Les contributions sont les bienvenues ! Si vous avez une idée ou une amélioration, ouvrez une pull request.
