

Plan de Projet RL Volatile Arms
## Introduction
Ce document présente le plan de projet pour l'implémentation d'une extension complexe
aux problèmes de Multi-Armed Bandits (MAB) et Contextual Multi-Armed Bandits (CMAB).
Notre équipe a choisi d'explorer l'extension des "Volatile Arms", où la disponibilité des bras
(actions) fluctue dynamiquement au fil du temps. Cette approche est inspirée du papier de
recherche de référence : Chen Lixing, Xu Jie et Lu Zhuo, NeurIPS 2018 1.
Justification du choix du dataset MovieLens
Pour rendrPour rendre ce projet réaliste et pertinent, nous utiliserons les fichiers de
données fournis :
arms.csv
## ,
contexts.csv
et
results.csv
. Ces fichiers représentent un scénario
métier où les bras (actions), leurs contextes et les résultats observés sont déjà structurés.
Cette approche nous permet de nous concentrer directement sur l'implémentation de la
volatilité et l'adaptation des algorithmes, en simulant un catalogue de films rotatif (licences
expirées, nouveaux ajouts) justifiant la volatilité.es.
Mise à jour prévue pour le simulateur
Le simulateur existant sera mis à jour pour intégrer la mécanique de volatilité. Cela
impliquera la création d'un "masque de disponibilité" dynamique à chaque itération
t
## . Ce
masque binaire indiquera quels bras sont actifs et disponibles pour la sélection par les
algorithmes à un instant donné. Le simulateur devra être capable de retourner cet état
partiel des bras disponibles aux algorithmes.
Plan d'implémentation algorithmique
Nous adapterons plusieurs algorithmes classiques de MAB et CMAB pour qu'ils puissent
gérer la volatilité des bras. Les algorithmes suivants seront modifiés pour filtrer les bras
indisponibles lors de l'étape de sélection (argmax) :
## •
UCB1 (Upper Confidence Bound 1)
## •
## Thompson Sampling
## •
LinUCB (Linear Upper Confidence Bound)
## •
LinTS (Linear Thompson Sampling)
L'évaluation de ces algorithmes sera réalisée sur 10 itérations (runs) indépendantes afin
d'assurer la robustesse des résultats et de mesurer le regret cumulé.

Répartition du travail
Le tableau ci-dessous récapitule la répartition des tâches entre les trois membres de
l'équipe, garantissant une charge de travail équilibrée et une spécialisation des rôles.
MembreRôleMissions
## Principales
Cibles Barème à
## Valider
## Livrables
## Attendus
## Membre 1
## Lead Data &
## Simulateur
Intégrer les
fichiers
arms.csv
## ,
contexts.csv
et
results.csv
## .
Mettre à jour
Mabs_Simulator

## (
data_loader.py
## ,
simulator.py
## )
pour la
volatilité
(masque de
disponibilité).
"Simulator
update" (6
points)
Code du
simulateur
capable de
retourner un
état partiel des
bras
disponibles.
## Membre 2
## Lead
## Algorithmique
Modifier les
algorithmes
(EGreedy,
Random, UCB1,
TS, LinUCB,
CTS) pour
respecter le
masque de
disponibilité.
Implémenter la
boucle de tests
(10 runs) et
extraire les
métriques
(regret cumulé).
## Garantie
absolue de la
norme PEP-8.
"Algorithm
implementation
" (6 points) +
"Code Quality"
(3 points)
## Fichiers
algorithmiques
adaptés et
fonctionnels.
Membre 3Lead
## Scientifique
Maîtriser le
papier NeurIPS
2018 sur les
## Volatile Arms.
Rédiger le
rapport final.
"Setting
## Complexity" (3
points) +
"Justifications
& explanations"
(4 points) +
Rapport final
structuré,
professionnel
et analytique.

## Références
[1] Chen Lixing, Xu Jie and Lu Zhuo. (2018). Multi-Armed Bandits with Volatile Arms.
NeurIPS 2018.
## Justifier
scientifiquemen
t le
comportement
des
algorithmes,
expliquer le
choix de
MovieLens,
analyser les
graphiques de
performances.
"Redaction
quality" (3
points)