# %% [markdown]
# # 01 — Analyse exploratoire : Credit Card Fraud Detection
#
# Objectifs de ce notebook (cf. `sujet.txt`, « Analyse exploratoire ») :
#
# 1. **Qualité des données** : types, valeurs manquantes, doublons.
# 2. **Caractériser le déséquilibre** et montrer pourquoi l'accuracy est trompeuse.
# 3. **Variables interprétables** `Amount` et `Time`.
# 4. **Variables anonymisées** `V1`–`V28` : structure ACP, pouvoir séparateur, valeurs extrêmes, échelles.
# 5. **Visualisation en faible dimension** (t-SNE) : les fraudes forment-elles un groupe ?
# 6. **Synthèse** et conséquences pour la modélisation.
#
# Les figures sont enregistrées dans `reports/figures/` pour le rapport.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import StandardScaler

from fraude.data import FEATURES, PCA_FEATURES, PROJECT_ROOT, TARGET, load_creditcard

SEED = 42
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
PALETTE = {0: "#4C72B0", 1: "#C44E52"}
LABELS = {0: "Légitime", 1: "Fraude"}

def save(fig, name):
    fig.savefig(FIG_DIR / f"eda_{name}.png", dpi=150, bbox_inches="tight")

df = load_creditcard()
df.shape

# %% [markdown]
# ## 1. Qualité des données

# %%
print("Types :", df.dtypes.value_counts().to_dict())
print("Valeurs manquantes :", int(df.isna().sum().sum()))
df.describe().T

# %% [markdown]
# ### 1.1 Doublons
#
# Des lignes strictement identiques posent un problème de **fuite d'information** : si une copie
# tombe dans l'échantillon d'entraînement et l'autre dans le test, le modèle est évalué sur une
# observation déjà vue. On vérifie aussi qu'aucun doublon n'a de classes contradictoires
# (mêmes variables, `Class` différente), ce qui signalerait du bruit d'étiquetage.

# %%
dup_mask = df.duplicated(keep="first")          # copies excédentaires
in_dup_group = df.duplicated(keep=False)        # toutes les lignes concernées
contradictory = (df.groupby(FEATURES)[TARGET].nunique() > 1).sum()

pd.Series({
    "Lignes excédentaires (copies)": int(dup_mask.sum()),
    "  dont fraudes": int(df.loc[dup_mask, TARGET].sum()),
    "Lignes appartenant à un groupe de doublons": int(in_dup_group.sum()),
    "Groupes à classes contradictoires": int(contradictory),
    "Taille après dédoublonnage": int((~dup_mask).sum()),
    "Fraudes après dédoublonnage": int(df.loc[~dup_mask, TARGET].sum()),
}).to_frame("valeur")

# %% [markdown]
# **Décision proposée** : supprimer les doublons **avant** le découpage train/test
# (`load_creditcard(drop_duplicates=True)`). Aucun doublon n'est contradictoire, donc on ne perd
# aucune information ; on supprime seulement un risque de fuite. Coût : 19 fraudes de moins (492 → 473).
#
# La suite de l'EDA est faite sur les données **dédoublonnées**, qui seront celles de la modélisation.

# %%
df = load_creditcard(drop_duplicates=True)
df.shape, int(df[TARGET].sum())

# %% [markdown]
# ## 2. Le déséquilibre des classes

# %%
counts = df[TARGET].value_counts().rename(index=LABELS)
pd.DataFrame({"effectif": counts, "proportion": counts / len(df)})

# %%
fig, ax = plt.subplots(figsize=(6, 3.5))
bars = ax.bar(counts.index, counts.values, color=[PALETTE[0], PALETTE[1]])
ax.set_yscale("log")
ax.bar_label(bars, labels=[f"{v:,}\n({v/len(df):.3%})" for v in counts.values])
ax.set_ylabel("Nombre de transactions (échelle log)")
ax.set_title(f"Ratio légitime / fraude ≈ {counts.iloc[0] / counts.iloc[1]:.0f} : 1")
ax.set_ylim(top=2e6)
save(fig, "desequilibre"); plt.show()

# %% [markdown]
# ### 2.1 Le paradoxe de l'accuracy
#
# Un « classifieur » trivial qui prédit **toujours « légitime »** :

# %%
y = df[TARGET]
y_trivial = np.zeros_like(y)
pd.Series({
    "accuracy": accuracy_score(y, y_trivial),
    "balanced accuracy": balanced_accuracy_score(y, y_trivial),
    "précision (fraude)": precision_score(y, y_trivial, zero_division=0),
    "rappel (fraude)": recall_score(y, y_trivial),
    "F1 (fraude)": f1_score(y, y_trivial),
}).to_frame("classifieur trivial")

# %% [markdown]
# Plus de 99,8 % d'accuracy sans détecter **aucune** fraude : l'accuracy ne mesure ici que la
# proportion de la classe majoritaire. Elle ne pourra pas servir de critère de sélection.
# La balanced accuracy (0,5) et surtout le rappel / la précision sur la classe fraude révèlent l'échec.
#
# ### 2.2 Conséquence statistique : peu de fraudes pour évaluer
#
# Avec un test stratifié de 20 %, il n'y aura qu'environ 95 fraudes. Le rappel estimé est alors
# très bruité : si le vrai rappel vaut 0,80, l'écart-type de son estimation est
# $\sqrt{0{,}8 \times 0{,}2 / n_{\text{fraudes}}}$.

# %%
n_test_fraud = round(0.2 * y.sum())
for true_recall in (0.6, 0.8, 0.9):
    sd = np.sqrt(true_recall * (1 - true_recall) / n_test_fraud)
    print(f"rappel vrai {true_recall:.1f} -> écart-type ≈ {sd:.3f} "
          f"(IC95 ≈ ±{1.96 * sd:.2f}) avec {n_test_fraud} fraudes en test ; "
          f"1 fraude = {1 / n_test_fraud:.1%} de rappel")

# %% [markdown]
# Des écarts de quelques points de rappel entre deux méthodes ne seront donc **pas significatifs**
# sur un seul découpage → il faudra répéter les expériences (validation croisée répétée) et
# rapporter la variabilité.

# %% [markdown]
# ## 3. Variables interprétables : `Amount` et `Time`
#
# ### 3.1 Montant

# %%
amount_stats = df.groupby(TARGET)["Amount"].describe(percentiles=[.25, .5, .75, .95, .99])
amount_stats.index = amount_stats.index.map(LABELS)
amount_stats["part montant = 0"] = df.groupby(TARGET)["Amount"].apply(lambda s: (s == 0).mean()).values
amount_stats["skewness"] = df.groupby(TARGET)["Amount"].skew().values
amount_stats.T

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for c in (0, 1):
    sub = df.loc[df[TARGET] == c, "Amount"]
    axes[0].hist(sub, bins=100, density=True, alpha=.6, color=PALETTE[c], label=LABELS[c],
                 range=(0, 2500))
    axes[1].hist(np.log1p(sub), bins=60, density=True, alpha=.6, color=PALETTE[c], label=LABELS[c])
axes[0].set(title="Montant brut (tronqué à 2 500)", xlabel="Amount", ylabel="densité")
axes[1].set(title="Montant après log(1 + x)", xlabel="log1p(Amount)")
for ax in axes: ax.legend()
save(fig, "amount"); plt.show()

# %%
total = df.groupby(TARGET)["Amount"].sum()
print(f"Montant total des fraudes : {total[1]:,.0f} € "
      f"({total[1] / total.sum():.3%} du volume, pour {y.mean():.3%} des transactions)")

# %% [markdown]
# **À retenir (montant)**
# - Distribution très asymétrique (skewness élevée) : une transformation `log1p` la rend bien plus
#   symétrique, utile pour les modèles linéaires et les méthodes fondées sur les distances (SMOTE, kNN).
# - Les fraudes ont une **médiane plus faible** mais une **moyenne plus élevée** : elles mélangent des
#   micro-montants (tests de carte ?) et des montants élevés. Elles ne dépassent jamais ~2 100 €.
# - La proportion de montants nuls est nettement plus forte chez les fraudes.
# - Le montant est disponible pour chaque fraude → il pourra servir de **coût d'un faux négatif**
#   dans l'analyse coût-bénéfice.
#
# ### 3.2 Temps
#
# `Time` est le nombre de secondes écoulées depuis la **première transaction** du jeu (sur 2 jours).
# On en déduit une « heure relative » modulo 24 h. On ne connaît pas l'heure réelle de début, mais
# le creux d'activité observé correspond très probablement à la nuit.

# %%
df["hour"] = (df["Time"] // 3600 % 24).astype(int)
by_hour = df.groupby("hour")[TARGET].agg(transactions="size", fraudes="sum", taux="mean")

fig, ax1 = plt.subplots(figsize=(11, 4))
ax1.bar(by_hour.index, by_hour["transactions"], color=PALETTE[0], alpha=.5, label="transactions")
ax1.set(xlabel="heure relative (modulo 24 h)", ylabel="nombre de transactions")
ax2 = ax1.twinx()
ax2.plot(by_hour.index, by_hour["taux"] * 100, color=PALETTE[1], marker="o", label="taux de fraude")
ax2.axhline(y.mean() * 100, color=PALETTE[1], ls="--", lw=1)
ax2.set_ylabel("taux de fraude (%)"); ax2.grid(False)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper right")
ax1.set_title("Volume et taux de fraude par heure (tirets : taux moyen)")
save(fig, "time"); plt.show()
by_hour.T

# %% [markdown]
# **À retenir (temps)** : le volume chute fortement pendant la « nuit » (heures 0–7), alors que le
# nombre de fraudes n'y baisse pas (48 fraudes à l'heure 2, contre une vingtaine en journée) → le
# **taux** de fraude y atteint 1 à 1,5 %, soit 6 à 8 fois la moyenne (0,17 %).
# `Time` brut est peu informatif (il est monotone sur 2 jours) ; une variable dérivée de type heure
# du jour l'est davantage. Attention cependant : avec 2 jours seulement, ce motif reste fragile.

# %% [markdown]
# ## 4. Variables anonymisées `V1`–`V28`
#
# ### 4.1 Structure héritée de l'ACP
#
# Selon la documentation du jeu, `V1`–`V28` sont les composantes principales d'une ACP réalisée par
# les auteurs. On s'attend donc à des variables **centrées**, **non corrélées** et de **variance
# décroissante**.

# %%
pca_stats = pd.DataFrame({"moyenne": df[PCA_FEATURES].mean(), "variance": df[PCA_FEATURES].var()})
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.bar(PCA_FEATURES, pca_stats["variance"], color="grey")
ax.set(title="Variance des composantes V1–V28", ylabel="variance")
ax.tick_params(axis="x", rotation=90)
save(fig, "variance_pca"); plt.show()
print(f"|moyenne| max : {pca_stats['moyenne'].abs().max():.2e}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, (title, sub) in zip(axes, [("Toutes transactions", df), ("Fraudes uniquement", df[df[TARGET] == 1])]):
    corr = sub[PCA_FEATURES].corr()
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .7})
    off = corr.values[~np.eye(len(corr), dtype=bool)]
    ax.set_title(f"{title}\n|corr| moyenne hors diagonale = {np.abs(off).mean():.2f}")
save(fig, "correlations"); plt.show()

# %% [markdown]
# - Sur l'ensemble, les corrélations sont quasi nulles (l'ACP a bien été faite sur tout le jeu,
#   probablement **avant tout découpage** : une forme de fuite impossible à corriger, à mentionner
#   dans le rapport).
# - **Chez les fraudes, de fortes corrélations apparaissent** : les fraudes se déplacent de façon
#   coordonnée sur plusieurs composantes (V1–V18 surtout). Elles occupent une région particulière
#   de l'espace, ce qui est plutôt favorable à SMOTE (interpolation entre fraudes voisines).
# - Les variances décroissent comme attendu : `V1`–`V28` ne sont **pas sur la même échelle**,
#   et `Amount` (écart-type ≈ 250) encore moins.
#
# ### 4.2 Pouvoir séparateur univarié
#
# Trois mesures complémentaires pour chaque variable :
# - **écart de moyennes standardisé** $|\bar x_{fraude} - \bar x_{légit}| / s$ ;
# - **statistique de Kolmogorov-Smirnov** (écart maximal entre les deux fonctions de répartition, entre 0 et 1) ;
# - **information mutuelle** avec `Class` (capte aussi les relations non monotones).

# %%
fraud, legit = df[df[TARGET] == 1], df[df[TARGET] == 0]
cols = FEATURES + ["hour"]
mi = mutual_info_classif(df[cols], y, discrete_features=[c == "hour" for c in cols],
                         random_state=SEED, n_jobs=-1)
sep = pd.DataFrame({
    "écart moy. std": [abs(fraud[c].mean() - legit[c].mean()) / df[c].std() for c in cols],
    "KS": [ks_2samp(fraud[c], legit[c]).statistic for c in cols],
    "info. mutuelle": mi,
}, index=cols).sort_values("KS", ascending=False)
# L'information mutuelle est bornée par l'entropie de Class (très faible car p ≈ 0,0017) :
# on l'exprime aussi en part de ce maximum (coefficient d'incertitude).
p = y.mean()
H_class = -(p * np.log(p) + (1 - p) * np.log(1 - p))
sep["IM / H(Class)"] = sep["info. mutuelle"] / H_class
print(f"Entropie de Class : {H_class:.4f} nats (borne supérieure de l'information mutuelle)")
sep.style.background_gradient(cmap="Reds").format("{:.3f}")

# %%
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.bar(sep.index, sep["KS"], color=[PALETTE[1] if v > .6 else "grey" for v in sep["KS"]])
ax.set(title="Statistique KS fraude vs légitime (rouge : KS > 0,6)", ylabel="KS")
ax.tick_params(axis="x", rotation=90)
save(fig, "ks"); plt.show()

# %%
top = sep.index[:8].tolist()
weak = sep.index[-4:].tolist()
fig, axes = plt.subplots(3, 4, figsize=(16, 9))
for ax, c in zip(axes.flat, top + weak):
    lo, hi = df[c].quantile([.001, .999])
    for k in (0, 1):
        sns.kdeplot(df.loc[df[TARGET] == k, c].clip(lo, hi), ax=ax, color=PALETTE[k],
                    fill=True, alpha=.35, common_norm=False, label=LABELS[k], warn_singular=False)
    ax.set(title=f"{c} (KS = {sep.loc[c, 'KS']:.2f})", xlabel="", ylabel="")
axes[0, 0].legend()
fig.suptitle("Densités par classe : 8 variables les plus séparatrices (2 premières lignes) "
             "et 4 les moins séparatrices", y=1.01)
fig.tight_layout(); save(fig, "densites"); plt.show()

# %% [markdown]
# **À retenir** : un petit groupe de variables (`V14`, `V10`, `V12`, `V4`, `V11`, `V17`, `V3`…) sépare
# très nettement les classes (KS > 0,7), souvent avec des fraudes déplacées vers des valeurs
# extrêmes (négatives pour V14, V12, V10, V17 ; positives pour V4, V11). À l'inverse `V13`, `V15`,
# `V22`, `V25`, `V26` ont des distributions presque identiques dans les deux classes. Ces
# résultats serviront de **référence pour l'explicabilité** : un modèle crédible devrait s'appuyer
# principalement sur les premières.
#
# ### 4.3 Valeurs extrêmes
#
# Règle de Tukey (au-delà de 1,5 × IQR, IQR calculé sur l'ensemble) : quelle part de chaque classe
# serait considérée comme « aberrante » ?

# %%
q1, q3 = df[FEATURES].quantile(.25), df[FEATURES].quantile(.75)
iqr = q3 - q1
is_out = (df[FEATURES] < q1 - 1.5 * iqr) | (df[FEATURES] > q3 + 1.5 * iqr)
out_rate = is_out.groupby(df[TARGET]).mean().T.rename(columns=LABELS)
out_rate["rapport fraude / légitime"] = out_rate["Fraude"] / out_rate["Légitime"]
print("Part des transactions avec ≥ 1 valeur extrême :",
      is_out.any(axis=1).groupby(df[TARGET]).mean().rename(LABELS).round(3).to_dict())
print("Part des transactions avec ≥ 1 valeur extrême sur les 8 variables les plus séparatrices :",
      is_out[top].any(axis=1).groupby(df[TARGET]).mean().rename(LABELS).round(3).to_dict())
out_rate.loc[top]

# %%
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, c in zip(axes, top[:4]):
    sns.boxplot(data=df, x=TARGET, y=c, hue=TARGET, palette=PALETTE, ax=ax, legend=False,
                flierprops={"markersize": 2, "alpha": .3})
    ax.set_xticks([0, 1], [LABELS[0], LABELS[1]]); ax.set_xlabel("")
fig.suptitle("Boîtes à moustaches par classe"); fig.tight_layout()
save(fig, "boxplots"); plt.show()

# %% [markdown]
# **À retenir** : 92 % des fraudes ont au moins une « valeur extrême » (au sens de Tukey) sur les
# 8 variables les plus séparatrices, contre 17 % des transactions légitimes. Sur toutes les
# variables, presque la moitié des transactions légitimes en ont une aussi : la règle de Tukey n'a
# pas grand sens sur ces distributions à queues lourdes. **Supprimer les outliers reviendrait à
# supprimer presque toutes les fraudes** : on les conserve. En revanche, pour les modèles sensibles à l'échelle, on
# préférera une standardisation robuste (`RobustScaler`, médiane / IQR) ou au moins `StandardScaler`
# ajusté uniquement sur le train.
#
# ### 4.4 Différences d'échelle

# %%
df[FEATURES].agg(["std", "min", "max"]).T.sort_values("std", ascending=False).head(10)

# %% [markdown]
# `Time` (écart-type ≈ 47 000) et `Amount` (≈ 250) dominent complètement les `V` (≈ 0,3 à 2) :
# sans mise à l'échelle, une distance euclidienne (kNN, SMOTE, ADASYN, NearMiss, ENN) serait
# quasiment déterminée par `Time` et `Amount`. **La standardisation doit précéder tout
# rééchantillonnage fondé sur les voisins**, dans le pipeline.

# %% [markdown]
# ## 5. Visualisation t-SNE
#
# Sur un sous-échantillon (toutes les fraudes + 5 000 transactions légitimes tirées au hasard),
# après standardisation. t-SNE préserve les **voisinages locaux** mais pas les distances globales :
# on regarde seulement si les fraudes sont regroupées ou dispersées parmi les légitimes.

# %%
sample = pd.concat([fraud, legit.sample(5000, random_state=SEED)])
Z = StandardScaler().fit_transform(sample[PCA_FEATURES + ["Amount"]].assign(Amount=np.log1p(sample["Amount"])))
emb = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca").fit_transform(Z)

fig, ax = plt.subplots(figsize=(8, 7))
for k in (0, 1):
    m = sample[TARGET].values == k
    ax.scatter(emb[m, 0], emb[m, 1], s=4 if k == 0 else 10, alpha=.4 if k == 0 else .8,
               color=PALETTE[k], label=f"{LABELS[k]} (n={m.sum():,})")
ax.set(title="t-SNE (V1–V28 + log1p(Amount), standardisées)", xticks=[], yticks=[])
ax.legend(markerscale=3)
save(fig, "tsne"); plt.show()

# %% [markdown]
# **Lecture** : une grande partie des fraudes forme un ou quelques amas bien distincts, mais une
# fraction se mélange aux transactions légitimes. On peut anticiper que :
# - les fraudes « isolées » seront faciles à détecter par la plupart des méthodes ;
# - la difficulté (et donc les différences entre méthodes de rééquilibrage) se jouera sur les
#   fraudes **mêlées aux légitimes**, là où SMOTE peut créer des points synthétiques dans des zones
#   légitimes et où ADASYN concentrera justement sa génération. Des méthodes de nettoyage (ENN,
#   Tomek) pourraient alors être utiles.
#
# (Rappel : le sous-échantillonnage gonfle artificiellement la proportion de fraudes sur la figure.)

# %% [markdown]
# ## 6. Synthèse et conséquences pour la modélisation
#
# | Constat | Conséquence |
# |---|---|
# | 1 081 doublons exacts (19 fraudes), aucun contradictoire | Dédoublonnage **avant** le split : 283 726 transactions, 473 fraudes |
# | 0,17 % de fraudes (≈ 600 : 1) ; le classifieur trivial a 99,8 % d'accuracy | Accuracy exclue comme critère ; PR-AUC, rappel, précision, F1/F2 |
# | ≈ 95 fraudes dans un test à 20 % ; 1 fraude ≈ 1 point de rappel | CV stratifiée répétée, rapporter moyenne ± écart-type |
# | `Amount` très asymétrique, fraudes ≤ ~2 100 € | `log1p(Amount)` ; montant utilisable comme coût d'un faux négatif |
# | Taux de fraude plus élevé la « nuit » | Variable dérivée `hour` à tester (prudence : 2 jours seulement) |
# | `V` issues d'une ACP sur tout le jeu (corrélation nulle globalement) | Limite à mentionner (fuite non corrigible) |
# | Corrélations fortes **au sein** des fraudes ; quelques variables très séparatrices (V14, V10, V12, V4, V17…) | Signal fort et concentré ; référence pour l'explicabilité |
# | Les fraudes sont majoritairement des valeurs extrêmes | **Ne pas** supprimer les outliers ; envisager `RobustScaler` |
# | Échelles très différentes (`Time`, `Amount` vs `V`) | Standardisation dans le pipeline, avant SMOTE/ADASYN/kNN |
# | t-SNE : fraudes en partie groupées, en partie mêlées aux légitimes | Comparer SMOTE vs ADASYN vs méthodes avec nettoyage |
