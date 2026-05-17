# api/main.py
# SenSante API - Assistant pré-diagnostic médical

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib 
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from groq import Groq

# Charger les variables d'environnement
load_dotenv()

# Client Groq (chargé au démarrage)
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialisé.")
else:
    print(
        "ATTENTION : GROQ_API_KEY non trouvée. "
        "/explain sera désactivé."
    )

class ExplainInput(BaseModel):
    diagnostic: str = Field(
        ...,
        description="Diagnostic prédit par le modèle"
    )

    probabilite: float = Field(
        ...,
        description="Probabilité du diagnostic"
    )

    age: int = Field(
        ...,
        description="Âge du patient"
    )

    sexe: str = Field(
        ...,
        description="Sexe du patient"
    )

    temperature: float = Field(
        ...,
        description="Température du patient"
    )

    region: str = Field(
        ...,
        description="Région du patient"
    )


class ExplainOutput(BaseModel):
    explication: str = Field(
        ...,
        description="Explication en français"
    )

    modele_llm: str = Field(
        default="llama-3.1-8b-instant",
        description="Modèle LLM utilisé"
    )

# --- Schemas Pydantic ---
class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool
    fatigue: bool
    maux_tete: bool
    region: str


class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str


# --- Application ---
app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.2.0"
)

# Autoriser les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Chargement du modèle ---
print("Chargement du modèle...")

model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

print(f"Modèle chargé : {list(model.classes_)}")


# --- Routes ---
@app.get("/")
def root():
    return {
        "message": "Bienvenue sur SenSante API",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model-info": "/model-info",
            "explain": "/explain"
        }
    }


@app.get("/favicon.ico")
def favicon():
    return {"message": "No favicon configured"}


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}


@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):

    # Encodage sexe
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}"
        )

    # Encodage région
    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Région inconnue : {patient.region}"
        )

    # Features
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]])

    # Prédiction
    diagnostic = model.predict(features)[0]
    proba_max = float(model.predict_proba(features)[0].max())

    # Confiance
    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    # Messages
    messages = {
        "palu": "Suspicion de paludisme. Consultez rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation.",
        "typhoide": "Suspicion de typhoïde. Consultation nécessaire.",
        "sain": "Pas de pathologie détectée."
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un médecin.")
    )

@app.get("/model-info")
def model_info():
    return {
        "type_modele": type(model).__name__,
        "nombre_arbres": model.n_estimators,
        "classes": list(model.classes_),
        "nombre_features": len(feature_cols)
    }

# Prompt système pour le LLM
SYSTEM_PROMPT = """
Tu es un médecin sénégalais. 
RÉPOND EXCLUSIVEMENT EN MÉLANGEANT WOLOF ET FRANÇAIS (Sénégalo-français).
C'est impératif.

Expressions obligatoires :
- Salutation : "As-salamu alaykum" ou "Nanga def"
- Fièvre : "Sa yaram dafa tchaat"
- Diagnostic : "Loolu mën na doon [diagnostic]"
- Conseil : "Faw nga fadjou dji"
- Conclusion : "Tanante"

Ne fais pas de phrases trop longues.
"""

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    """Expliquer un diagnostic en français avec un LLM."""

    if not groq_client:
        return ExplainOutput(
            explication=(
                "Service d'explication indisponible. "
                "Clé API non configurée."
            ),
            modele_llm="aucun"
        )

    # Construire le prompt utilisateur
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, "
        f"région {data.region}\n"
        f"Température : {data.temperature}°C\n"
        f"Diagnostic du modèle : {data.diagnostic} "
        f"(probabilité {data.probabilite:.0%})\n"
        f"Explique ce résultat au patient."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens=200,
            temperature=0.3
        )

        explication = response.choices[0].message.content

    except Exception as e:
        explication = (
            f"Erreur lors de l'appel au LLM : {str(e)}"
        )

    return ExplainOutput(
        explication=explication,
        modele_llm="llama-3.3-70b-versatile"
    )