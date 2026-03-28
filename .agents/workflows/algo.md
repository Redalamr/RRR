---
description: Exécuteur Gate 1 - Crée le fichier Python d'un algorithme MAB en se basant sur les papiers de recherche.
---

# /algo [Nom_de_l_algorithme]

## Rôle
Tu es l'**Exécuteur Gate 1 (Création)** du projet MAB Simulator pour l'utilisateur **Reda**.

## Contexte du Projet
- **Code source** : `C:\Users\redlam\Music\prj_RR\Mabs_Simulator`
- **Papiers algorithmes** : `C:\Users\redlam\Music\prj_RR\algoo`
- **Algorithmes existants** : `C:\Users\redlam\Music\prj_RR\Mabs_Simulator\Src\algorithms\`

## Tâche
1. Identifier le papier de recherche correspondant à l'algorithme demandé dans `C:\Users\redlam\Music\prj_RR\algoo\` :
   - UCB → `UCB (1).md`
   - TS → `TS.md`
   - LinUCB → `LinUCB.md`
   - LinTS → `LinTS.md`
2. Lire **intégralement** le fichier Markdown du papier de recherche.
3. Étudier la structure des algorithmes existants (`EGreedy.py`, `Random.py`) pour respecter le même pattern de code (classes, méthodes, signatures).
4. Créer le fichier Python de l'algorithme dans `C:\Users\redlam\Music\prj_RR\Mabs_Simulator\Src\algorithms\`.
5. Mettre à jour `C:\Users\redlam\Music\prj_RR\Mabs_Simulator\whatedited.md`.

## Contraintes Professeur IMPÉRATIVES
- **UCB** : Implémenter uniquement **UCB1**. Pas UCB2, KL-UCB ou autre variante.
- **TS (Thompson Sampling)** : Première implémentation avec des **récompenses de Bernoulli ({0,1})** uniquement. L'adaptation pour récompenses continues [0,1] se fera dans un second temps.
- **LinUCB** : Modèle **"disjoint"** uniquement. Pas le modèle hybride.
- **LinTS** : Ne commencer que **APRÈS** que LinUCB soit terminé et validé.

## Règles de Code STRICTES
1. **ZÉRO commentaire** `#` dans le code. AUCUN. Ni en ligne, ni au-dessus d'une ligne.
2. **Documentation** : Utiliser uniquement des **Docstrings Python** (`""" """`) sous les déclarations de classes et de méthodes.
3. **Simplicité** : Code simple, direct, académique. Pas de sur-ingénierie.
4. **Un seul algorithme par invocation** de cette commande.
5. **Respecter le pattern** des fichiers existants (EGreedy.py, Random.py) pour la structure de classe et les signatures de méthodes.

## Avant de coder
- Lire `EGreedy.py` et `Random.py` pour comprendre :
  - La classe de base / interface utilisée
  - Les méthodes attendues (`select_arm`, `update`, etc.)
  - Le format des paramètres
  - La structure des imports

## Output Attendu
- Un fichier Python unique dans `Src/algorithms/` respectant toutes les contraintes.
- Mise à jour de `whatedited.md` avec la date, le nom du fichier et les détails.
