========================================================================
PROJET DE CLASSIFICATION D'IMAGES : CHIENS ET CHATS (DEEP LEARNING)
========================================================================

Auteur : Dominique Fotsing Souop
Environnement : PyTorch, Torchvision, Gradio, Google Colab (GPU Tesla T4), Visual Studio Code


1. DESCRIPTION DU PROJET
------------------------------------------------------------------------
Ce projet met en œuvre un pipeline complet de vision par ordinateur pour la classification binaire d'images (Chien vs Chat) à l'aide du transfert d'apprentissage (Transfer Learning). Le système comprend la phase d'entraînement et d'évaluation, la sauvegarde du meilleur modèle, ainsi qu'une interface utilisateur web interactive développée avec Gradio.


2. STRUCTURE DU PROJET
------------------------------------------------------------------------
├── Projet_Classification_Chien_Chat.ipynb
│   └── Notebook Jupyter gérant la configuration Google Drive/GPU, le chargement des données (DataLoader, batch size 32), la visualisation des images et l'entraînement.
│
├── Class_chien_chat.py
│   └── Script d'inférence et déploiement de l'interface Gradio (chargement du modèle, prétraitement et prédiction en temps réel).
│
├── meilleur_modele_resnet18.pt/meilleur_modele_Adam.pt/meilleur_modele_Sgd.pt (NON INCLUS DANS LE DOSSIER A CAUSE DE LEURS TAILLES)
│   └── Fichier de poids du modèle entraîné sauvegardé.
│
└── my_helper.py
|   └── Fichier des dépendances.
│
└── Requirements.txt
    └── Fichier des dépendances.    

3. INSTALLATION
------------------------------------------------------------------------
git clone https://github.com/fotsingsouop/classification-chien-chat.git

cd classification-chien-chat


4. GUIDE D'UTILISATION
------------------------------------------------------------------------
1. Entraînement :
   - Ouvrez le notebook `Projet_Classification_Chien_Chat.ipynb` dans Google Colab.
   - Assurez vous que le fichier `Projet_Classification_Chien_Chat.ipynb` est dans le dossier /content/drive/MyDrive/Deep_learning/cnn_cat_dog_image_classification.
   - Assurez vous que le fichier `my_helper.py` est dans le dossier /content/drive/MyDrive/Deep_learning/cnn_cat_dog_image_classification
   - Assurez-vous que le dossier /content/drive/MyDrive/Deep_learning/cnn_cat_dog_image_classification contient le dossier Cat_Dog_data avec tous ses fichiers (les fichiers sont disponibles a ce lien https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fc%2Fdogs-vs-cats).
   - Assurez-vous d'activer l'accélérateur matériel GPU (T4).
   - Exécutez les cellules pour entraîner le modèle et générer `meilleur_modele_resnet18.pt`.

2. Lancement de l'interface Web (Gradio) :
   - Ouvrer Visual studio code.
   - Ouvrir le dossier du projet.
   - Exécutez la commande suivante dans le terminal:

     pip install -r Requirements.txt

   - Assurez-vous que `meilleur_modele_resnet18.pt` se trouve dans le même répertoire que `Class_chien_chat.py`.
   - Exécutez la commande suivante dans le terminal:

     python Class_chien_chat.py

   - Accédez à l'interface via le lien local (ex: http://127.0.0.1:7860) ou l'URL publique générée par Gradio.