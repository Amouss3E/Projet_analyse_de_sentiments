import streamlit as st
from deep_translator import GoogleTranslator
from langdetect import detect
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import matplotlib.pyplot as plt
import nltk
import os

# Configurer un répertoire personnalisé pour NLTK
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Télécharger les ressources nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)

# Nettoyage du texte
stop_words = set(stopwords.words('english'))  # Liste des stopwords standard
custom_stopwords = {"mc", "mcdonald", "mcdonalds", "mc donald", "mc donalds"}  # Stopwords personnalisés
stop_words.update(custom_stopwords)  # Ajout des stopwords personnalisés

def clean_text(text):
    text = text.lower()  # Passage en minuscule
    text = re.sub(r'[^a-z\s]', '', text)  # Suppression des caractères spéciaux
    tokens = word_tokenize(text)  # Tokenisation
    tokens = [word for word in tokens if word not in stop_words]  # Suppression des stopwords
    return ' '.join(tokens)

# Charger le modèle entraîné et le vectoriseur
model = joblib.load('LGBMClassifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer_LGBMC.pkl')

# Interface Streamlit
st.title("Démonstration de prédiction de sentiment")

# Description des données d'entraînement
st.write("""
**Description des données d'entraînement :**
L'ensemble de données utilisé pour l'entraînement de notre modèle contient plus de 33 000 avis anonymisés sur les magasins McDonald's aux États-Unis, extraits des avis Google. Il fournit des informations sur les expériences et les opinions des clients sur divers établissements McDonald's à travers le pays.
""")

# Exemples de textes par label et par langue
st.write("""
**Exemples de textes :**

**Français :**
- **Negative** : "Le service était lent et la nourriture était froide."
- **Neutral** : "Le restaurant était propre, mais la nourriture était moyenne."
- **Positive** : "Ce restaurant était propre et le service était rapide."

**Anglais :**
- **Negative** : "The food was cold and the service was terrible."
- **Neutral** : "The restaurant was clean, but the food was average."
- **Positive** : "I loved this place! The food was delicious and the staff was friendly."
""")

# Entrée utilisateur
user_input = st.text_area("Entrez un texte en français ou en anglais :")

# Bouton pour prédire le sentiment
if st.button("Prédire le sentiment"):
    if user_input:
        try:
            # Détecter la langue du texte avec langdetect
            detected_lang = detect(user_input)
            st.write(f"Langue détectée : {detected_lang}")
            
            # Traduire le texte en fonction de la langue détectée
            if detected_lang == 'fr':
                translated_text = GoogleTranslator(source='fr', target='en').translate(user_input)
                st.write(f"Texte traduit en anglais : {translated_text}")
            elif detected_lang == 'en':
                translated_text = GoogleTranslator(source='en', target='fr').translate(user_input)
                st.write(f"Texte traduit en français : {translated_text}")
            else:
                st.warning("Langue non prise en charge. Veuillez entrer un texte en français ou en anglais.")
                translated_text = user_input  # Utiliser le texte original si la langue n'est pas prise en charge
            
            # Nettoyer le texte
            cleaned_text = clean_text(translated_text if detected_lang in ['fr', 'en'] else user_input)
            
            # Vectoriser le texte
            text_vectorized = vectorizer.transform([cleaned_text])
            
            # Prédire le sentiment
            prediction = model.predict(text_vectorized)
            prediction_proba = model.predict_proba(text_vectorized)
            
            # Afficher le résultat en français
            sentiment_map = {
                'Negative': 'Négatif',
                'Neutral': 'Neutre',
                'Positive': 'Positif'
            }
            predicted_sentiment_fr = sentiment_map.get(prediction[0], prediction[0])
            st.write(f"Sentiment prédit : **{predicted_sentiment_fr}**")
            
            # Afficher les pourcentages de prédiction
            st.write("**Pourcentages de prédiction pour chaque label :**")
            labels = model.classes_
            proba = prediction_proba[0]
            
            # Traduire les labels en français pour le diagramme
            labels_fr = [sentiment_map.get(label, label) for label in labels]
            
            # Créer un diagramme circulaire
            fig, ax = plt.subplots()
            ax.pie(proba, labels=labels_fr, autopct='%1.1f%%', colors=['red', 'blue', 'green'])
            ax.set_title('Répartition des sentiments')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Une erreur s'est produite : {e}")
    else:
        st.write("Veuillez entrer un texte pour prédire le sentiment.")
