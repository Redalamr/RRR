# Historique des Modifications - MAB Simulator

## 2026-03-27

### whatedited.md
- Création du fichier de suivi des modifications du projet.

### UCB1.py
- Création du fichier `Src/algorithms/UCB1.py`.
- Implémentation de l'algorithme UCB1 (Upper Confidence Bound 1) basé sur Auer, Cesa-Bianchi & Fischer (2002).
- Formule de sélection : x̄_i + √(2 ln(t) / n_i).
- Respecte le pattern de classe existant (EGreedy.py, Random.py) : __init__, run, init_choice, choose_action, evaluate, update.

### TS.py
- Création du fichier `Src/algorithms/TS.py`.
- Implémentation de Thompson Sampling pour récompenses Bernoulli basé sur Agrawal & Goyal (2012).
- Échantillonne θ_i ~ Beta(S_i + 1, F_i + 1) pour chaque bras, joue argmax θ_i.
- Respecte le pattern de classe existant (EGreedy.py, Random.py) : __init__, run, init_choice, choose_action, evaluate, update.

### LinUCB.py
- Création du fichier `Src/algorithms/LinUCB.py`.
- Implémentation de LinUCB modèle disjoint basé sur Li et al. (2010), Algorithm 1.
- Récompenses Bernoulli (seuil ≥ 4 → reward = 1, sinon 0).
- Régression ridge par bras : A_a (matrice d×d, init identité), b_a (vecteur d, init zéro).
- Sélection UCB : θ̂_a^T x + α √(x^T A_a^{-1} x).
- Mise à jour : A_a ← A_a + x x^T, b_a ← b_a + r · x.
- Respecte le pattern de classe existant : __init__, run, init_choice, choose_action, evaluate, update.

### LinTS.py
- Création du fichier `Src/algorithms/LinTS.py`.
- Implémentation de Thompson Sampling contextuel basé sur Agrawal & Goyal (2013), Algorithm 1.
- Récompenses Bernoulli (seuil ≥ 4 → reward = 1, sinon 0).
- Paramètre partagé μ : B (matrice d×d, init identité), f (vecteur d, init zéro), μ̂ = B⁻¹f.
- Échantillonne μ̃ ~ N(μ̂, v² B⁻¹), joue argmax b_i^T μ̃.
- Mise à jour : B ← B + x xᵀ, f ← f + r · x, μ̂ ← B⁻¹f.
- Respecte le pattern de classe existant : __init__, run, init_choice, choose_action, evaluate, update.

### simulator.py
- Ajout de l'import de `UCB1` depuis `Src.algorithms.UCB1`.
- Ajout de l'import de `TS` depuis `Src.algorithms.TS`.
- Ajout de l'import de `LinUCB` depuis `Src.algorithms.LinUCB`.
- Ajout de l'import de `LinTS` depuis `Src.algorithms.LinTS`.
- **Task 5** : Modification de `__init__` pour accepter `algorithm_name` (défaut `"EGreedy"`) et `d` (dimension contexte).
  - Ajout d'un dictionnaire de classe `ALGORITHM_MAP` qui associe chaque nom à sa classe.
  - Dispatch automatique : `LinUCB` et `LinTS` reçoivent le paramètre `d` ; les autres (EGreedy, Random, UCB1, TS) reçoivent uniquement `arms`.
  - `ValueError` levée si le nom est inconnu ou si `d` est manquant pour LinUCB/LinTS.
  - Suppression de tous les commentaires `#` du fichier (conformité règles de code).
- **Task 13** : Ajout de la méthode `generate_availability_mask(self, arms, iteration)`.
  - Retourne un `np.ndarray` binaire (0/1) de longueur `len(arms)` indiquant les bras actifs.
  - Probabilité de base par bras tirée dans [0.5, 0.95] (seed 123, reproductible).
  - Modulation cyclique sinusoïdale (période propre par bras, entre 50 et 500 itérations).
  - Probabilité effective = clip(base + 0.2 × sin(2π × iteration / period), 0.05, 1.0).
  - Garantie qu'au moins un bras est toujours disponible.
  - Ajout de `import pandas as pd` et nettoyage des séparateurs `#-----------------------`.
- **Task 14** : Dans `run_simulation`, intégration du masque de disponibilité.
  - Après le tirage de `user_id`, appel à `generate_availability_mask` pour obtenir les bras actifs.
  - Filtrage de `observed_value` pour ne conserver que les bras actifs (`arm_id` présents dans le masque).
  - La note brute (1 à 5) est envoyée telle quelle à `algorithm.run()`, `algorithm.update()` et `results.update_measures()`, la binarisation restant gérée en interne par chaque algorithme via `evaluate()`.

### EGreedy.py
- **Task 15** : Vérification du filtrage des bras actifs.
  - `init_choice()` filtre déjà `arms_pool` via `isin(observation["arm_id"])`.
  - `choose_action()` itère uniquement sur `self.arms_pool` (phases initialisation et exploitation) — aucune modification nécessaire.
  - Suppression de tous les commentaires `#` et ajout de docstring de classe.

### Random.py
- **Task 16** : Vérification du filtrage des bras actifs.
  - `init_choice()` filtre déjà `arms_pool` via `isin(observation["arm_id"])`.
  - `choose_action()` tire au sort uniquement parmi `self.arms_pool.index` — aucune modification nécessaire.
  - Ajout de `self.threshold = 4` pour cohérence avec les autres algorithmes.
  - Suppression de tous les commentaires `#` et ajout de docstrings.

### UCB1.py
- **Task 17** : Vérification filtrage argmax + binarisation dans `update()`.
  - `choose_action()` calcule les valeurs UCB uniquement sur `self.arms_pool` — déjà correct.
  - `update()` modifiée : binarisation explicite `reward_binaire = 1 if feedback >= self.threshold else 0` sans appeler `evaluate()`.
  - `evaluate()` non modifiée.

### TS.py
- **Task 18** : Vérification sampling Beta + binarisation dans `update()`.
  - `choose_action()` échantillonne theta uniquement pour les bras de `self.arms_pool` — déjà correct.
  - `update()` modifiée : binarisation explicite avec `self.threshold` avant mise à jour successes/failures, sans appeler `evaluate()`.
  - `evaluate()` non modifiée.

### LinUCB.py
- **Task 19** : Vérification calcul UCB + binarisation dans `update()`.
  - `choose_action()` calcule les valeurs UCB uniquement sur `self.arms_pool` — déjà correct.
  - `update()` modifiée : binarisation explicite `reward_binaire = 1 if feedback >= self.threshold else 0` avant mise à jour de `b[arm]`, sans appeler `evaluate()`.
  - `evaluate()` non modifiée.

### LinTS.py
- **Task 20** : Vérification sampling + binarisation dans `update()`.
  - `choose_action()` échantillonne et fait l'argmax uniquement sur `self.arms_pool` — déjà correct.
  - `update()` modifiée : binarisation explicite avec `self.threshold` avant mise à jour de `f` et `mu_hat`, sans appeler `evaluate()`.
  - `evaluate()` non modifiée.

### simulator.py
- **Task 21** : Ajout de la méthode `run_multiple_simulations(self, nb_runs=10)`.
  - Exécute `run_simulation` N fois de manière indépendante.
  - À chaque run, réinstancie l'algorithme (frais, sans état) et crée un nouveau `ResultStorer`.
  - Réinitialise le RNG de disponibilité des bras avec un seed différent par run (`42 + run_index`).
  - Collecte le regret cumulé final de chaque run dans un `np.ndarray`.
  - Calcule et affiche la moyenne et l'écart-type du regret cumulé sur les N runs.
  - Retourne un dict `{"all_cumulated_regrets", "mean_regret", "std_regret"}`.
  - Logique d'évaluation existante (`evaluate()`, `update_measures()`) non modifiée.

### results_storer.py
- **Task 22** : Adaptation de `ResultStorer` pour supporter le stockage multi-runs.
  - Ajout de l'attribut `self.horizon` pour permettre la réinitialisation des tableaux.
  - Ajout de l'attribut `self.all_runs_performance` (liste de dicts) pour stocker les métriques de chaque run.
  - Ajout de la méthode `reset_current_run()` : sauvegarde le run courant dans la liste puis réinitialise les tableaux.
  - Ajout de la méthode `store_current_run()` : sauvegarde le run courant sans réinitialiser.
  - Ajout de la méthode `compute_statistics()` : calcule moyennes et écarts-types de correctness, accuracy et cumulated_regrets sur les N runs (retourne un dict).
  - Suppression de tous les commentaires `#` et séparateurs, ajout de docstrings conformes.

### report_generator.py
- **Task 23** : Modification de la génération de rapports pour afficher les courbes de regret cumulé moyennées.
  - Ajout des imports `numpy`, `matplotlib` (backend `Agg` pour génération sans interface graphique).
  - Ajout de la méthode `plot_averaged_regret(self, statistics, algorithm_name)`.
  - Trace la courbe du regret cumulé moyen sur N runs avec `ax.plot()`.
  - Ajoute une enveloppe ombrée ± 1 écart-type via `ax.fill_between()` (alpha=0.25).
  - Titre dynamique incluant le nom de l'algorithme et le nombre de runs.
  - Sauvegarde du graphique en PNG (150 dpi) dans le dossier `results/` de l'output.
  - Suppression de tous les commentaires `#` et séparateurs du fichier original, ajout de docstrings conformes.

### simulator.py
- **Task 24** : Ajout du paramètre `dataset_name` à `Simulator.__init__` (défaut `"02-Mushrooms"`).
  - Permet de spécifier le dataset depuis `main.py` sans modifier le simulateur.
  - `self.dataset_name` est désormais initialisé avec la valeur du paramètre.

### main.py
- **Tasks 24, 25, 26** : Configuration complète du point d'entrée pour le dataset MovieLens.
  - Constantes : `DATASET_NAME = "06-MovieLens"`, `CONTEXT_DIM = 22`, `NB_RUNS = 10`.
  - Boucle sur les 4 algorithmes MAB (EGreedy, Random, UCB1, TS) puis les 2 CMAB (LinUCB, LinTS avec `d=22`).
  - Chaque algorithme instancie un `Simulator` dédié et appelle `run_multiple_simulations(nb_runs=10)`.
  - Le masque de volatilité est activé automatiquement (déjà intégré dans `run_simulation`).
  - Suppression de tous les commentaires `#` et séparateurs, ajout de docstring conforme.

### simulator.py
- **Task 26.1** : Correction de l'erreur de dimension pour les algorithmes contextuels.
  - Slicing de `user_context` dans `run_simulation()` (`user_context = user_context[1:]`) pour retirer `context_id` (1ère colonne) avant l'appel à `self.algorithm.run()`.
  - Assure la conformité avec la dimension `d=22` attendue pour MovieLens.
  - Nettoyage final garantissant l'absence de tout commentaire `#` (règle de style).
