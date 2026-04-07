"""
SenSante - Exploration du dataset patient_dakar.csv
lab 1 : Git, Python et structure du projet
"""
import pandas as pd

# Charger le dataset
df = pd.read_csv('data/patients_dakar.csv - patients_dakar.csv')

#  Premier APER US 
print("=" * 50)
print("SENSANTE - EXPLORATION DU DATASET")
print("=" * 50)

# Dimensions du dataset
print(f"\nNombre de patients : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"Colonnes : {list(df.columns)}")

# Aperçu des 5 premières lignes
print("\nAperçu des 5 premières lignes :")
print(df.head())

# Statistiques de base
print("\nStatistiques descriptives :")
print(df.describe().round(2))

# Repartition des diagnostics
print("\nRépartition des diagnostics :")
diag_counts = df['diagnostic'].value_counts()
for diag, count in diag_counts.items():
    pct = (count / len(df)) * 100
    print(f"  {diag:12s} : {count:3d} patients ({pct:.1f}%)")

# Repartition par region
print("\nRépartition des patients par région (top 5) :")
region_counts = df['region'].value_counts().head(5)
for region, count in region_counts.items():
    pct = (count / len(df)) * 100
    print(f"  {region:15s} : {count:3d} patients ")

# Temperature moyenne par diagnostic

print("\nTempérature moyenne par diagnostic :")
temp_by_diag = df.groupby('diagnostic')['temperature'].mean()
for diag, temp in temp_by_diag.items():
    print(f"  {diag:12s} : {temp:.1f} °C")

print("\n{'=' * 50}")
print("FIN DE L'EXPLORATION") 
print("Prochaine lab : Entrainer un modèle ML")
print(f"{'=' * 50}")
