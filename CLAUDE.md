# Projet : détection de fraude bancaire sur données déséquilibrées

Projet d'étude (méthodologie). Sujet complet : `sujet.txt`. Objectif : comparer les
méthodes de rééquilibrage (over/under-sampling, SMOTE, ADASYN, hybrides, pondération
des classes) et les critères d'évaluation adaptés, sur le jeu Kaggle
« Credit Card Fraud Detection » (`mlg-ulb/creditcardfraud`).
L'enjeu n'est pas le meilleur score mais **comprendre et justifier** les choix.

Langue du projet : **français** (code commenté, notebooks, rapport).

## Environnement
- Onyxia / SSP Cloud, Python 3.13. Package installé en mode éditable : `pip install -e .`
- Code réutilisable dans `src/fraude/`. Notebooks dans `notebooks/` au **format script `# %%`**
  (cellules interactives VS Code / Jupytext, `# %% [markdown]` pour le texte) — pas de `.ipynb`.
- Données : `data/` est dans `.gitignore` (+ `*.csv`, `*.parquet`, `*.zip`). **Ne jamais versionner les données.**

## Données
- `from fraude.data import load_creditcard, split_features_target`
- `load_creditcard()` : télécharge via kagglehub (pas d'identifiants requis), cache
  `data/processed/creditcard.parquet`, valide forme/colonnes/nb de fraudes.
- `python -m fraude.data` : téléchargement + résumé.
- 284 807 transactions, 492 fraudes (0,173 %, ratio ≈ 578:1), 31 colonnes :
  `Time` (s, sur 48 h), `V1`–`V28` (composantes ACP anonymisées), `Amount`, `Class` (1 = fraude).
- 1 081 lignes dupliquées (19 fraudes), aucune à classe contradictoire. `load_creditcard(drop_duplicates=True)`
  → 283 726 transactions, 473 fraudes. **C'est la version utilisée pour toute la modélisation.**

## Règles méthodologiques (à respecter dans tout le code)
- Découpage train/test **stratifié, avant tout rééchantillonnage**, graine fixe.
- Rééchantillonnage uniquement dans un `imblearn.pipeline.Pipeline`
  (scaler → sampler → modèle) pour qu'il ne s'applique qu'aux plis d'entraînement.
- Validation : CV stratifiée répétée sur le train ; test final touché une seule fois.
- Critère principal : **PR-AUC** (average precision). Accuracy = descriptif seulement.
  Aussi : précision, rappel, F1, F2, balanced accuracy, matrice de confusion, ROC-AUC.
- Ne pas supposer le seuil 0,5 optimal ; le seuil se choisit sur la validation, pas sur le test.
- Petit nombre de fraudes en test (~98 pour 20 %) → toujours rapporter la variabilité.
- Graines (`random_state`) fixées partout pour la reproductibilité.

## Plan d'étude
1. **EDA** (`notebooks/01_eda.py`)
   - Qualité : types, NA, doublons (dont doublons en fraude / à classes contradictoires).
   - Déséquilibre : baseline « tout légitime » → 99,83 % accuracy, rappel 0.
   - `Amount` (asymétrie, log1p, comparaison par classe), `Time` → heure du jour, taux de fraude par heure.
   - V1–V28 : corrélations globales vs sur fraudes seules ; pouvoir séparateur
     (écart de moyennes standardisé, KS, information mutuelle) ; densités par classe ;
     valeurs extrêmes (ne pas les supprimer aveuglément : le signal est dans les queues).
   - t-SNE/UMAP sur sous-échantillon (toutes fraudes + ~5 000 légitimes).
2. **Protocole** (`src/fraude/benchmark.py`) : split, CV, pipelines, calcul des métriques par pli.
3. **Premières expériences** : LogReg + Random Forest × {sans traitement, class_weight,
   RandomOverSampler, SMOTE, RandomUnderSampler} ; effet de `sampling_strategy` (0,01 → 1).
   Hypothèse à tester : le rééquilibrage déplace surtout le seuil effectif plutôt que la
   capacité de classement (PR-AUC proches, F1@0,5 très différents).
4. **Extensions** : ADASYN, BorderlineSMOTE, Tomek, ENN, NearMiss, SMOTEENN, SMOTETomek ;
   gradient boosting (`scale_pos_weight`).
5. **Seuil & coûts** : courbes précision/rappel vs seuil, seuil F-beta ;
   coût FN = `Amount` de la fraude ratée, coût FP = coût fixe de contrôle (sensibilité à faire varier).
6. **Robustesse** : variance inter-plis / inter-graines. **Explicabilité** : SHAP, cohérence avec l'EDA.
7. **Livrables** : rapport méthodologique, notebooks reproductibles, figures (courbes PR,
   matrices de confusion), recommandations, éventuelle démo interactive du seuil.

## Avancement
- [x] Module d'import des données (`src/fraude/data.py`)
- [x] Plan d'étude
- [x] EDA (`notebooks/01_eda.py`, figures dans `reports/figures/eda_*.png`)
- [ ] Protocole de benchmark
- [ ] Premières expériences
- [ ] Extensions, seuil/coûts, robustesse, explicabilité
- [ ] Rapport

## Décisions prises
<!-- Consigner ici chaque choix méthodologique et sa justification (date). -->
- 2026-11-20 : doublons supprimés **avant** le split (risque de fuite train/test, aucune perte
  d'information car aucun doublon contradictoire). Loader : `drop_duplicates=True`.
- 2026-11-20 : valeurs extrêmes **conservées** (92 % des fraudes ont une valeur extrême sur les
  8 variables les plus séparatrices, contre 17 % des légitimes).
- 2026-11-20 : `log1p(Amount)` ; standardisation dans le pipeline avant tout sampler à base de voisins.

## Constats clés de l'EDA (pour le rapport et l'interprétation)
- Classifieur trivial : accuracy 0,998, balanced accuracy 0,5, rappel 0.
- ≈ 95 fraudes en test à 20 % → 1 fraude ≈ 1,1 pt de rappel ; IC95 du rappel ≈ ±0,08.
- `Amount` : fraudes de médiane plus faible (9,8 contre 22) mais de moyenne plus forte, max 2 126 € ;
  5,3 % de montants nuls (contre 0,6 %). Total des fraudes ≈ 58 600 €.
- `hour` (heure relative = Time // 3600 % 24) : taux de fraude de 1 à 1,5 % aux heures 2 à 4 (la nuit), contre 0,17 % en moyenne.
- V1–V28 : décorrélées globalement (ACP faite sur tout le jeu → fuite non corrigible, à citer),
  mais fortement corrélées **au sein des fraudes** (|corr| moyenne 0,30).
- Variables les plus séparatrices (KS) : V14 0,84 ; V10 0,80 ; V12 0,78 ; V4 0,76 ; V11 0,75 ;
  V17 0,74 ; V3 0,70 ; V16 0,68. Les moins séparatrices : V13, V15, V22, V25, V26 (KS < 0,11).
- t-SNE : un gros amas de fraudes bien séparé + une fraction de fraudes dispersées parmi les légitimes.
