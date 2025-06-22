# POC-RAG-System: Chatbot Culturel Occitanie

Ce projet implémente un **chatbot culturel** capable de recommander des événements en Occitanie en se basant sur des données provenant d'Open Agenda, et d'interagir avec l'utilisateur de manière naturelle. Le chatbot utilise **FAISS** pour l'indexation des événements et **MistralAI** pour les embeddings et la génération de réponses via un modèle de langage.

## Fonctionnalités

1. **Mémoire conversationnelle** intégrée
2. **Géolocalisation** automatique via IP, fallback sur Toulouse
3. **Recherche web en direct** via DuckDuckGo lorsqu’aucune donnée n’est trouvée localement
4. **Monitoring & feedback** :
   - Logs horodatés dans `csv/chatbot_logs_<YYYY-MM-DD>.csv`
   - Question, ville, fallback, temps de réponse
   - Feedback global `o/n` après fin de session
5. **Gestion de la date "du jour"** :
   - Injection de la `current_date` dans le prompt
   - (Optionnel) Support du parsing intelligent ("ce weekend" → dates réelles) via `python-dateparser`


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

Les tests unitaires sont exécutés automatiquement à chaque push grâce à GitHub Actions.

Pour exécuter les tests localement :
   - python -m unittest discover tests/

## Exemples d’utilisation

Voici quelques exemples de requêtes que vous pouvez tester avec le chatbot :

- "Quels concerts sont prévus à Toulouse ce week-end ?"
- "Y a-t-il des spectacles à Montpellier entre le 10 et le 20 juillet 2025 ?"
- "Quels événements originaux recommandes-tu pour bientôt ?"

Le chatbot tentera d’interpréter les expressions temporelles floues et vous répondra à l’aide des données les plus pertinentes disponibles.


## Structure du Projet

      ├── chatbot-culturel-occitanie/
         ├── chatbot.py           
         ├── geo.py #module de géolocalisation     
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
