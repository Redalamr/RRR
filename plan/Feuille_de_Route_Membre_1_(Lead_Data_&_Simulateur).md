

Feuille de Route : Membre 1 (Lead Data &
## Simulateur)
## Rôle
## Lead Data & Simulateur
## Missions
Le Membre 1 est responsable de l'intégration et de la préparation du dataset, ainsi que de la
mise à jour du simulateur pour gérer la volatilité des bras.
## •
Intégration et Nettoyage des fichiers de données (arms.csv, contexts.csv,
results.csv) :
## •
Charger les fichiers
arms.csv
## ,
contexts.csv
et
results.csv
dans le projet.
## •
Effectuer les opérations de nettoyage et de prétraitement nécessaires pour assurer
la qualité des données.
## •
Préparer le dataset pour être utilisé par le simulateur et les algorithmes.
## •
Mise à jour du
Mabs_Simulator
## Python :
## •
Modifier les fichiers
data_loader.py
et
simulator.py
pour y injecter la mécanique de
volatilité.
## •
Développer une fonction capable de générer un "masque de disponibilité" binaire
pour les bras à chaque itération
t
## .
## •
S'assurer que le simulateur peut communiquer cet état partiel des bras disponibles
aux algorithmes de MAB/CMAB.
Cible Barème à Valider
## •
"Simulator update" (6 points)
## Livrables Attendus
## •
Code du simulateur capable de retourner un état partiel des bras disponibles aux
algorithmes.
## •
Scripts d'intégration et de nettoyage des fichiers de données.