import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import gradio as gr

# ==============================================================================
# 1. CHARGEMENT ET PRÉPARATION DU MODÈLE
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def charger_modele_resnet(save_path="meilleur_modele_resnet18.pt"):
    """Reconstruit la structure ResNet18 et charge vos poids entraînés."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 2)
    )
    
    # Chargement des poids enregistrés
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # Passer en mode évaluation (désactive le Dropout)
    return model

# Chargement du modèle au démarrage de l'application
model = charger_modele_resnet("meilleur_modele_resnet18.pt")


# Pipeline de prétraitement officiel ImageNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

"""
# Pipeline avec conservation du ratio d'aspect (Resize 256 + CenterCrop 224)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
"""

# ==============================================================================
# 2. FONCTION DE PRÉDICTION COMPATIBLE GRADIO
# ==============================================================================
def classifier_image(image_pil):
    """
    Prend une image PIL en entrée et retourne un dictionnaire {Classe: Probabilité}
    compatible avec le composant gr.Label.
    """
    if image_pil is None:
        return None
    
    # conversion en RGB pour gérer les images PNG ou N&B
    image_pil = image_pil.convert("RGB")
    
    # Transformation et ajout de la dimension du batch
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    
    # Inférence
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilites = F.softmax(outputs, dim=1)[0]
    
    classes = ['Chat', 'Chien']
    
    # Gradio attend un dictionnaire de type {'Chat': 0.965, 'Chien': 0.035}
    return {classes[i]: float(probabilites[i]) for i in range(len(classes))}

# ==============================================================================
# 3. CRÉATION ET LANCEMENT DE L'INTERFACE WEB
# ==============================================================================
demo = gr.Interface(
    fn=classifier_image,
    inputs=gr.Image(type="pil", label="Glissez-déposez une image ici"),
    outputs=gr.Label(num_top_classes=2, label="Résultat du Modèle"),
    title="🐱 Classifier Chien vs Chat (ResNet18) 🐶",
    description="Application Web alimentée par PyTorch et ResNet18 (Précision : 96.56%). Importez une photo pour tester la prédiction.",
    theme="soft"
)

# Lancement de l'interface locale
if __name__ == "__main__":
    # Passer share=True pour générer un lien public temporaire accessible depuis votre téléphone
    demo.launch(share=False)