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

## Protocole (`src/fraude/benchmark.py`)
- `load_split()` : 80/20 stratifié, graine 42 → train 226 980 (378 fraudes), test 56 746 (95 fraudes).
- `load_temporal_split()` : jour 1 → jour 2. Train 144 236 (272 fraudes, 0,19 %),
  test 139 490 (201 fraudes, 0,14 %) : **le taux de fraude baisse au jour 2**.
- `make_pipeline(model, sampler=None, use_hour=False)` : log1p(Amount) + StandardScaler,
  puis sampler, puis modèle. `Time` brut retiré ; `use_hour=True` ajoute l'heure en sin/cos.
- `evaluate(pipeline, X, y, name)` : 5 plis × 3 répétitions, **mêmes plis pour toutes les
  méthodes** (comparaisons appariées), en parallèle. Renvoie un `CVResult` (`metrics` : une ligne
  par pli, avec la taille de l'échantillon vu après rééchantillonnage ; `predictions` : index de
  la transaction, y_true, proba, pour chaque répétition). Sauvegardé dans `data/results/<name>/`,
  rechargeable avec `CVResult.load(name)`.
- `summarize` / `format_summary` : tableau « moyenne ± écart-type ».
- `evaluate_on_test` : test final uniquement (ou découpage temporel).
- `quick_evaluate(pipeline, X, y)` : mode exploratoire, **1 seul pli** (exactement le pli 0 de
  `evaluate`), non sauvegardé par défaut. PR-AUC à ± 0,05 près : tout écart plus petit doit être
  confirmé par `evaluate`. Mettre `n_jobs=-1` dans le modèle dans ce mode.
- Pas de GPU sur le pod, et scikit-learn n'en utilise pas de toute façon.
- **HistGradientBoosting : toujours mettre `l2_regularization` (ex. 1.0).** Sans régularisation,
  il diverge sur ces données (la PR-AUC chute de 0,76 après 1 itération à 0,14 après 100 itérations,
  y compris sur l'entraînement). Cause probable : les feuilles contenant quelques fraudes ont une
  hessienne p(1−p) ≈ 0, donc des pas de Newton énormes. Avec l2 = 1 : PR-AUC 0,84 sur le pli 0.
- Le pod a un **quota de 15 CPU** (`nproc` affiche 128, mais ce sont les cœurs de la machine hôte). `evaluate` lance
  déjà 15 plis en parallèle → mettre `n_jobs=1` dans les modèles (sinon surcharge). Temps indicatifs :
  régression logistique ≈ 5 s ; Random Forest 200 arbres ≈ 4 à 7 min (≈ 1,3 s par arbre sur 180 000 lignes).

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

## Pistes prioritaires (choisies par l'utilisateur, 2026-11-23)
Elles passent avant les extensions de l'étape 4. Le protocole doit permettre de les traiter
sans réentraîner : **conserver les probabilités prédites de chaque pli, avec l'index de chaque
transaction**.
1. **Découpage temporel ou découpage aléatoire.** Entraîner sur le jour 1 (Time < 86 400 s) et
   tester sur le jour 2, puis comparer au découpage stratifié aléatoire. Objectif : savoir si
   les performances et le classement des méthodes tiennent dans le temps. Limite : un seul
   découpage temporel possible (2 jours), donc pas de variabilité. À traiter comme analyse de
   robustesse, pas comme protocole principal.
2. **Typologie des fraudes « faciles » et « difficiles ».** Pour chaque fraude, calculer la
   proportion de fraudes parmi ses k plus proches voisins (k = 5) dans l'espace standardisé,
   puis la classer « sûre », « en frontière », « rare » ou « isolée » (Napierała et Stefanowski,
   2016). Ensuite, mesurer **sur quelle catégorie chaque méthode de rééquilibrage fait gagner
   (ou perdre) du rappel**, et où elle crée des faux positifs. Objectif : expliquer *pourquoi*
   SMOTE, ADASYN ou ENN marchent ou non, au lieu de seulement comparer des scores.
   Point de départ : le t-SNE de l'EDA (un amas de fraudes isolé, d'autres fraudes noyées
   parmi les légitimes).

Autres pistes notées, moins prioritaires : calibration et correction des probabilités après
sous-échantillonnage (Dal Pozzolo et al., 2015) ; coût par transaction fondé sur `Amount` ;
tests statistiques appariés entre méthodes (test t corrigé de Nadeau et Bengio, Wilcoxon).

## Avancement
- [x] Module d'import des données (`src/fraude/data.py`)
- [x] Plan d'étude
- [x] EDA (`notebooks/01_eda.py`, figures dans `reports/figures/eda_*.png`)
- [x] Protocole de benchmark (`src/fraude/benchmark.py`)
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
- 2026-11-23 : `Time` brut **retiré** des variables (compteur monotone sur 2 jours, inutilisable
  en découpage temporel). L'heure relative (sin/cos) est une option, `use_hour=True`, à évaluer.
  Premier essai en régression logistique : aucun effet (PR-AUC 0,756 contre 0,755).

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
