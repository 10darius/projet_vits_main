# Rapport sur les Variables de Configuration de l'Entraînement

Ce document détaille le rôle des paramètres clés présents dans la section `"train"` du fichier de configuration JSON pour le modèle VITS.

---

### Paramètres de Durée et de Fréquence

#### `epochs` (Époques)
- **Ce que c'est :** Le nombre total de fois où le modèle va parcourir l'intégralité du jeu de données d'entraînement.
- **Son rôle :** C'est la durée totale de l'apprentissage. Plus ce nombre est élevé, plus le modèle a de temps pour s'améliorer et converger vers un résultat optimal.

#### `eval_interval` (Intervalle d'Évaluation et de Sauvegarde)
- **Ce que c'est :** La fréquence, en nombre d'itérations (`steps`), à laquelle le modèle est sauvegardé et évalué.
- **Son rôle :** Ce paramètre est crucial pour suivre la progression. À chaque intervalle, le système :
    1.  Sauvegarde un point de contrôle (`checkpoint`) du modèle (`G_xxxx.pth` et `D_xxxx.pth`).
    2.  Génère des échantillons audio de test, que vous pouvez écouter dans l'onglet "AUDIO" de TensorBoard.
    3.  Calcule les métriques sur le jeu de données de validation.

#### `log_interval` (Intervalle des Logs)
- **Ce que c'est :** La fréquence (en itérations) à laquelle les informations sur la perte (loss) sont affichées dans la console et enregistrées dans TensorBoard.
- **Son rôle :** Permet de suivre la "santé" de l'entraînement en temps quasi réel.

---

### Paramètres de Performance et de Mémoire

#### `batch_size` (Taille du Lot)
- **Ce que c'est :** Le nombre de "segments" d'audio traités simultanément par le modèle à chaque itération.
- **Son rôle :** Une taille de lot plus grande peut rendre l'apprentissage plus stable et rapide. Cependant, c'est le paramètre qui a le **plus grand impact sur la consommation de mémoire** (RAM et VRAM). Une valeur trop élevée sur un système non adapté provoquera une erreur "not enough memory".

#### `segment_size` (Taille du Segment)
- **Ce que c'est :** La taille (en nombre d'échantillons audio) de chaque "morceau" d'audio que le modèle analyse à chaque étape.
- **Son rôle :** Une plus grande taille permet au modèle de voir plus de contexte, l'aidant à mieux apprendre les mélodies, la prosodie et les intonations sur de plus longues durées. Cela a également un impact significatif sur l'utilisation de la mémoire.

#### `use_gpu` (Utiliser le GPU)
- **Ce que c'est :** Une option (`true` ou `false`) pour dire au script d'utiliser la carte graphique pour les calculs.
- **Son rôle :** L'entraînement sur un GPU est des dizaines, voire des centaines de fois plus rapide que sur le processeur (CPU). C'est un paramètre essentiel pour un entraînement efficace.

#### `fp16_run` (Entraînement en Précision Mixte)
- **Ce que c'est :** Si mis à `true`, active l'entraînement en "précision mixte", qui utilise des nombres à virgule flottante de 16 bits au lieu des 32 bits standard.
- **Son rôle :** Sur les cartes graphiques NVIDIA récentes (dotées de Tensor Cores), cela peut accélérer considérablement l'entraînement et réduire l'utilisation de la mémoire GPU. Nous le laissons à `false` pour l'instant par souci de stabilité.

---

### Paramètres de l'Apprentissage (Optimiseur)

#### `learning_rate` (Taux d'Apprentissage)
- **Ce que c'est :** La "taille des pas" que le modèle effectue pour corriger ses erreurs à chaque itération.
- **Son rôle :** C'est un des réglages les plus critiques. S'il est trop élevé, le modèle "dépasse" la solution et n'apprend pas. S'il est trop faible, il apprend extrêmement lentement. La valeur de `2e-4` (0.0002) est un excellent point de départ standard pour cette architecture.

#### `lr_decay` (Décroissance du Taux d'Apprentissage)
- **Ce que c'est :** Un facteur qui réduit très légèrement le `learning_rate` après chaque époque.
- **Son rôle :** C'est une technique classique qui permet au modèle de faire de grands "pas" d'apprentissage au début, puis des pas de plus en plus fins pour peaufiner les résultats à mesure qu'il s'améliore.

#### `betas` et `eps`
- **Leur rôle :** Ce sont des paramètres de configuration avancés pour l'optimiseur `AdamW`, qui est l'algorithme qui met à jour les poids du modèle. Les valeurs par défaut sont basées sur les recommandations des chercheurs qui ont créé cet algorithme et sont presque universellement utilisées.

---

### Paramètres de la Fonction de Perte (Objectifs d'Apprentissage)

Le modèle VITS a plusieurs objectifs (erreurs ou "pertes") à minimiser simultanément. Ces coefficients permettent de doser l'importance de chaque objectif.

#### `c_mel` (Coefficient de la Perte Mel-spectrogramme)
- **Son rôle :** Contrôle l'importance pour le modèle de générer un spectrogramme qui soit le plus proche possible de celui de l'audio réel. C'est crucial pour la clarté et la fidélité du son.

#### `c_kl` (Coefficient de la Perte de Divergence KL)
- **Son rôle :** Un paramètre fondamental du "Variational Autoencoder" (VAE) de VITS. Il encourage le modèle à organiser de manière cohérente l'information qu'il apprend à partir du texte, ce qui est essentiel pour lui permettre de synthétiser la parole correctement pendant l'inférence (quand il n'a que du texte en entrée).
