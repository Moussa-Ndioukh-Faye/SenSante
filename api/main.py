# api/ main .py
# API FastAPI pour SenSante - Assistant pre - diagnostic medical
from fastapi import FastAPI
# Creer l' application
app = FastAPI (
title =" SenSante API",
description =" Assistant pre - diagnostic medical pour le Senegal ",
version =" 0.2.0 "
)
# Route de base : verifier que l'API fonctionne
@app . get ("/ health ")
def health_check () :

#Verification de l'etat de l'API.
    return {
    " status ": "ok",
    " message ": " SenSante API is running "
    }