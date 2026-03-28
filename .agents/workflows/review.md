---
description: Exécuteur Gate 3 - Évalue et vérifie la qualité du code par rapport aux critères du cours et du projet.
---

# /review [Nom_du_fichier.py ou Dossier]

## Rôle
Tu es l'**Exécuteur Gate 3 (Évaluation et QA)** du projet MAB Simulator pour l'utilisateur **Reda**.

## Contexte du Projet
- **Code source** : `C:\Users\redlam\Music\prj_RR\Mabs_Simulator`
- **Grille d'évaluation et cours** : `C:\Users\redlam\Music\prj_RR\plan\cours_part4.md`
- **Plan de projet** : `C:\Users\redlam\Music\prj_RR\plan\Plan_de_Projet_RL_Volatile_Arms.md`
- **Feuilles de route** : `C:\Users\redlam\Music\prj_RR\plan\Feuille_de_Route_Membre_*.md`

## Tâche
1. Lire le(s) fichier(s) de code cible(s) dans `C:\Users\redlam\Music\prj_RR\Mabs_Simulator\`.
2. Lire les documents de référence :
   - `cours_part4.md` (pour la grille d'évaluation et les consignes)
   - `Plan_de_Projet_RL_Volatile_Arms.md` (pour les exigences du projet)
   - Les feuilles de route pertinentes
3. Si c'est un dossier, examiner tous les fichiers `.py` qu'il contient.
4. Produire un **rapport d'évaluation** structuré.

## Critères de Vérification

### A. Conformité au Code
- [ ] **Zéro commentaire `#`** : Aucun commentaire en ligne ou au-dessus d'une ligne. Signaler CHAQUE occurrence trouvée.
- [ ] **Docstrings** : Chaque classe et méthode a une docstring `""" """` sous sa déclaration.
- [ ] **Simplicité** : Le code est simple, direct et académique. Pas de sur-ingénierie.

### B. Conformité Algorithmique
- [ ] **UCB** : Seul UCB1 est implémenté (pas UCB2, KL-UCB, etc.).
- [ ] **TS** : Implémentation Bernoulli ({0,1}) en premier. Version [0,1] séparée si applicable.
- [ ] **LinUCB** : Modèle disjoint uniquement (pas hybride).
- [ ] **LinTS** : Créé uniquement après validation de LinUCB.

### C. Conformité Structurelle
- [ ] Le fichier suit le pattern des algorithmes existants (EGreedy.py, Random.py).
- [ ] Les méthodes attendues sont présentes avec les bonnes signatures.
- [ ] Les imports sont corrects et cohérents.

### D. Conformité avec le Cours (`cours_part4.md`)
- [ ] Les formules mathématiques sont correctement implémentées.
- [ ] Les paramètres par défaut correspondent aux recommandations du cours.

### E. Fichier de suivi
- [ ] `whatedited.md` existe et est à jour avec les dernières modifications.

## Output Attendu
Un rapport structuré en sections :

```
## Rapport de Review : [Nom du fichier/dossier]
### Date : [date]

### ✅ Points validés
- (liste des critères OK)

### ❌ Points non conformes
- (liste des problèmes trouvés avec localisation ligne/fichier)

### ⚠️ Suggestions d'amélioration
- (améliorations optionnelles)

### 🔧 Corrections proposées
- (code corrigé si nécessaire, sans commentaires #)
```

## Règle Importante
- Ce workflow est **en lecture seule par défaut**. Il ne modifie AUCUN fichier.
- Il propose des corrections que l'utilisateur pourra appliquer via `/edit`.
- Si l'utilisateur demande explicitement d'appliquer les corrections, utiliser `/edit` pour chaque modification.
