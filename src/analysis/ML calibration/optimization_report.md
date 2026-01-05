# Rapport d'Optimisation du Réseau de Neurones pour la Calibration

## Résumé Exécutif

L'objectif était d'optimiser un réseau de neurones pour prédire les corrections de phase à partir de cartes de phase (calibration). Nous avons itéré sur l'architecture, la fonction de perte et les hyperparamètres en utilisant un dataset existant de 2000 échantillons pour minimiser le temps de calcul.

**Meilleure Performance (Test MSE) :** `2.82` (MLP sans Batch Normalization)
**Baseline (CNN) :** `3.07`
**Pire Performance :** `3.36` (CNN + CosineLoss)

## Méthodologie

Nous avons utilisé le script `neural_calibration.py` modifié pour :
1.  Utiliser un dataset pré-généré (`dataset_ns2000_nst20_gamma9.npz`).
2.  Supprimer les visualisations graphiques pour accélérer les itérations.
3.  Tester différentes configurations (MLP vs CNN, Loss Functions, Normalization).

## Résultats des Itérations

| Itération | Configuration | Train MSE | Test MSE | Observations |
|-----------|---------------|-----------|----------|--------------|
| **Baseline** | CNN, CosineLoss | N/A | 3.07 | Modèle de base. |
| **1** | MLP, UnitCircleMSELoss | 0.04 | 3.22 | Fort overfitting. La régularisation n'aide pas la généralisation. |
| **2** | MLP, Tanh, Dropout 0.3 | 0.22 | 3.20 | Moins d'overfitting mais pas d'amélioration sur le test. |
| **3** | Simple MLP, Low LR | 0.07 | 3.14 | Performance similaire à l'aléatoire (~3.29). |
| **4** | **MLP, No Batch Norm** | **0.05** | **2.82** | **Meilleur résultat.** La suppression de la BN aide la généralisation. |
| **5** | MLP, No BN, Dropout 0.1 | 0.09 | 3.02 | Moins bon que Dropout 0.05. |
| **6** | Large MLP, No BN, No Drop | 0.01 | 3.16 | Overfitting massif (Train MSE très bas, Test MSE élevé). |
| **7** | Tiny MLP (64-32), No Drop | 1.08 | 3.19 | Underfitting (Train MSE élevé). Trop simple pour apprendre. |

## Analyse

1.  **Problème de Généralisation (Distribution Shift) :**
    *   Le modèle apprend très bien sur le jeu d'entraînement (Train MSE ~0.01 - 0.05) qui est constitué de phases **aléatoires**.
    *   Il échoue sur le jeu de test (Test MSE ~2.82 - 3.16) qui est constitué de phases **constantes** (tous les shifters à la même valeur).
    *   Cela suggère que le modèle a appris à interpoler dans le nuage de points aléatoires mais ne comprend pas la physique sous-jacente pour extrapoler à des cas structurés (phases plates).

2.  **Impact de la Taille du Réseau :**
    *   Un réseau trop petit (Tiny MLP) est en **sous-apprentissage** (Train MSE > 1.0) et ne parvient même pas à mémoriser le dataset d'entraînement.
    *   Un réseau trop grand (Large MLP) est en **sur-apprentissage** immédiat.
    *   L'optimum semble être un MLP de taille moyenne (512-256 neurones).

3.  **Impact de la Batch Normalization :**
    *   La suppression de la Batch Normalization a apporté la plus grande amélioration.
    *   Cela confirme que les statistiques (moyenne/variance) des activations sont très différentes entre le jeu d'entraînement (aléatoire) et le jeu de test (structuré), rendant la BN nuisible.

## Recommandations

1.  **Enrichir le Dataset (CRITIQUE) :** Il est impératif d'inclure des cas structurés (phases constantes, gradients, etc.) dans le jeu d'entraînement, et pas seulement du bruit blanc uniforme. Le modèle n'a jamais vu de "phases plates" pendant l'entraînement.
2.  **Éviter la Batch Normalization :** Pour ce type de problème où la distribution de test peut être singulière par rapport au train, la BN est risquée.
3.  **Architecture :** Continuer avec un MLP de taille moyenne (512-256) avec un Dropout modéré (0.05). Éviter les architectures trop simplistes ("Tiny") qui sous-apprennent.

## Prochaines Étapes Suggérées

*   Générer un nouveau dataset mixte (50% aléatoire, 50% structuré).
*   Tester des méthodes non-neuronales (Random Forest) pour voir si elles capturent mieux les corrélations simples.
*   Utiliser le script avec l'option `--no-plot` pour les entraînements batch.
