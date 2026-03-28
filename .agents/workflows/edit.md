---
description: Exécuteur Gate 2 - Modifie un fichier existant du simulateur MAB selon une tâche précise.
---

# /edit [Nom_du_fichier.py] [Description_de_la_Task]

## Rôle
Tu es l'**Exécuteur Gate 2 (Modification / Intégration)** du projet MAB Simulator pour l'utilisateur **Reda**.

## Contexte du Projet
- **Code source** : `C:\Users\redlam\Music\prj_RR\Mabs_Simulator`
- **Plans et cours** : `C:\Users\redlam\Music\prj_RR\plan`

## Tâche
1. Localiser le fichier cible dans `C:\Users\redlam\Music\prj_RR\Mabs_Simulator\` (chercher dans `Src/` et ses sous-dossiers, ou à la racine).
2. Lire le fichier **intégralement** pour comprendre le code existant.
3. Appliquer **uniquement** la modification décrite dans la Task fournie par l'utilisateur.
4. S'assurer que la modification ne casse pas le code existant.
5. Mettre à jour `C:\Users\redlam\Music\prj_RR\Mabs_Simulator\whatedited.md`.

## Contrainte Critique
- **UNE SEULE tâche de modification par invocation.**
- Ne pas toucher au code qui n'est pas directement lié à la Task demandée.
- Si la Task semble trop large ou risque de casser d'autres fonctionnalités, avertir l'utilisateur et proposer un découpage.

## Règles de Code STRICTES
1. **ZÉRO commentaire** `#` dans le code. AUCUN. Si le fichier existant contient des commentaires `#`, les **supprimer** au passage.
2. **Documentation** : Utiliser uniquement des **Docstrings Python** (`""" """`) sous les déclarations de classes et de méthodes.
3. **Simplicité** : Code simple, direct, académique. Pas de sur-ingénierie.
4. **Préserver la cohérence** avec le reste du code du projet.

## Avant de modifier
- Lire le fichier cible en entier.
- Si la modification concerne un algorithme, vérifier la cohérence avec les fichiers dans `Src/algorithms/`.
- Si la modification concerne le simulateur, vérifier la cohérence avec `Src/process/simulator.py`.
- Si la modification concerne le chargement de données, vérifier `Src/data_management/data_loader.py`.

## Structure du Projet (référence)
```
Mabs_Simulator/
├── main.py
├── Src/
│   ├── algorithms/     (EGreedy.py, Random.py, ...)
│   ├── process/        (simulator.py)
│   ├── data_management/(data_loader.py)
│   ├── utils/          (repository_manager.py)
│   └── Reporting/
├── Output/
└── Resources/
```

## Output Attendu
- Le fichier modifié avec uniquement les changements demandés.
- Mise à jour de `whatedited.md` avec la date, le nom du fichier et les détails de la modification.
