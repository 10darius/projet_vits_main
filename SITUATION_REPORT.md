# Rapport de Situation Actuel du Projet VITS Multilingue

Ce rapport résume la situation actuelle du projet VITS Multilingue, incluant les problèmes rencontrés, les solutions apportées, les limitations identifiées et la feuille de route pour les prochaines étapes.

## 1. Contexte du Projet

L'objectif est d'adapter le modèle VITS (Variational Inference with adversarial learning for Text-to-Speech) pour la synthèse vocale multilingue, en se concentrant sur le Français, le Ghomala' et l'Anglais. Le projet se base sur une version modifiée de `vits_multilingual-main`.

## 2. Problèmes Rencontrés et Solutions Apportées

Le cheminement jusqu'à l'entraînement du modèle a été ponctué de plusieurs défis techniques, résolus comme suit :

| Problème Rencontré | Solution Apportée | Impact |
| :----------------- | :---------------- | :----- |
| `FileNotFoundError` (chemins audio incorrects) | Détection de l'incohérence des chemins audio (par ex. `dataset_bbj/wav/dataset_bbj/wav/`).<br>**Correction des `audio_prefix`** dans `preprocess_multilingual_data_v2.py` pour chaque langue, afin d'assurer une construction correcte des chemins absolus vers les fichiers audio. | Résolution des erreurs de chemin dans les filelists générées. |
| Fichiers audio manquants/inaccessibles non détectés par `os.path.exists()` | `os.path.exists()` s'est avéré non fiable sur Windows pour certains fichiers manquants.<br>**Implémentation d'une vérification robuste** utilisant `os.path.getsize()` dans un bloc `try-except` dans `preprocess_multilingual_data_v2.py`. | Les filelists générées excluent désormais les références aux fichiers audio réellement manquants ou inaccessibles, évitant les `FileNotFoundError` fatales pendant l'entraînement. |
| `ghomala_lexicon.tsv` vide | Le fichier `ghomala_lexicon.tsv` était un placeholder vide.<br>**Peuplement du fichier `ghomala_lexicon.tsv`** avec les entrées pertinentes extraites de `ghomala-2025-11-26-18_34_57.csv`. | Le lexique Ghomala' est désormais utilisé, réduisant (mais n'éliminant pas) le recours au fallback caractère par caractère. |
| `ERROR: Phoneme 'é' from G2P is not in the symbol set.` | Le jeu de symboles (`symbols_multilingual_v2.py`) ne contenait pas les caractères français accentués (ex: 'é').<br>**Ajout des caractères français accentués** (y compris la cédille) à la liste `_latin_alphabet` dans `symbols_multilingual_v2.py`. | Le modèle peut désormais traiter correctement les textes français incluant ces caractères. |
| `train.py: error: required argument -m/--model` | L'argument `--model` était manquant lors de l'appel du script `train.py`.<br>**Ajout de l'argument `--model`** (ex: `vits_multilingual_fr_gh_en`) à la commande d'entraînement. | Le script `train.py` peut désormais s'initialiser correctement et créer les répertoires de logs. |
| `train.py: error: unrecognized arguments: --data_path` | Les arguments `--data_path` et `--val_data_path` ne sont pas attendus en ligne de commande par `train.py`.<br>**Suppression des arguments `--data_path` et `--val_data_path`** de la ligne de commande, car les chemins sont lus depuis le fichier `config.json`. | Le script `train.py` démarre correctement le processus d'initialisation. |
| `NameError: name 'TextAudioSpeakerLoader' is not defined` | Les classes `TextAudioSpeakerLoader` et `TextAudioSpeakerCollate` n'étaient pas importées dans `train.py`.<br>**Ajout des instructions `import` manquantes** pour ces classes dans `train.py`. | Résolution de l'erreur d'importation des classes de chargement de données. |
| `AttributeError: 'WN' object has no attribute 'cond_layer'` | Le module `WN` (utilisé pour la prédiction de durée et l'encodeur postérieur) tentait d'accéder à `self.cond_layer` alors que `gin_channels` était `0`, empêchant sa création.<br>**Ajout de `gin_channels: 192`** à la section `model` de `vits_multilingual-main/configs/multilingual.json`. | Le modèle `SynthesizerTrn` est désormais correctement initialisé avec les dimensions d'embedding du locuteur nécessaires. |
| `NameError: name 'y_hat' is not defined` | Erreur de syntaxe due à un saut de ligne (`\n`) inattendu dans l'unpacking des valeurs de retour de `net_g` dans `train.py`.<br>**Correction de l'erreur de syntaxe** en supprimant le saut de ligne. | L'unpacking des valeurs de retour de `net_g` se déroule correctement. |
| `AttributeError: 'SynthesizerTrn' object has no attribute 'emb_g'` | La couche d'embedding du locuteur (`self.emb_g`) n'était pas initialisée si `n_speakers` est `1`.<br>**Modification de la condition** dans `SynthesizerTrn.forward` et `infer` de `if self.n_speakers > 0:` à `if self.n_speakers > 1:`. | Correction de l'erreur d'attribut pour l'entraînement monolingue. |
| `RuntimeError: not enough memory` (CPU) | Erreur d'allocation mémoire sur CPU due à la taille des tenseurs.<br>**Réduction de `segment_size`** dans la configuration (de 8192 à 4096). | Tentative de réduction de la consommation mémoire. Le problème persiste et bloque l'entraînement sur CPU. |
| Environnement Python incohérent (`RuntimeError: stft requires return_complex`) | Le `RuntimeError` suggérait une incompatibilité de version PyTorch.<br>**Exécution explicite de toutes les commandes** Python via `conda run -n mon_env39 python ...` pour assurer l'utilisation de l'environnement Conda `mon_env39` spécifié. | Assure la cohérence de l'environnement et l'utilisation des dépendances compatibles. |

## 3. Limitations Actuelles

*   **Entraînement CPU bloqué par la mémoire**: Le modèle VITS ne peut pas être entraîné efficacement sur CPU en raison de limitations de mémoire (même après réduction de `segment_size` et `batch_size`). Cela empêche la progression de l'entraînement sans accès à un GPU.
*   **Couverture du Lexique Ghomala'**: Malgré le peuplement de `ghomala_lexicon.tsv`, de nombreux mots Ghomala' ne sont pas encore couverts, ce qui conduit à un "fallback" basé sur les caractères. Pour une langue tonale comme le Ghomala', cette approche n'est pas idéale et peut impacter la qualité de la synthèse vocale.
*   **Absence d'analyse de données détaillée**: Bien que les problèmes de chemin et d'existence de fichiers aient été corrigés, une analyse plus approfondie des datasets (qualité, distribution, équilibre des langues) pourrait être nécessaire.
*   **Hyperparamètres non optimisés**: Les hyperparamètres du modèle sont actuellement ceux par défaut/hérités et ne sont pas encore optimisés spécifiquement pour la tâche multilingue.

## 4. Feuille de Route et Prochaines Étapes

1.  **Priorité absolue : Accès et configuration d'un GPU**: L'entraînement du modèle VITS est bloqué sans ressources GPU. Cette étape est cruciale avant de pouvoir continuer. Une fois le GPU configuré, assurez-vous que PyTorch le détecte correctement (ex: `torch.cuda.is_available()` doit retourner `True`).
2.  **Exécuter les entraînements comparatifs**: Une fois le GPU disponible, lancer les entraînements monolingues (Français, Anglais, Ghomala' séparément) et l'entraînement multilingue (3 langues simultanément) comme détaillé dans `TRAINING_PLAN.md`.
3.  **Surveiller et Analyser l'Entraînement**: Suivre l'évolution des pertes et des métriques pour chaque modèle.
4.  **Enrichir le Lexique Ghomala'**: Continuer à étendre `ghomala_lexicon.tsv` pour augmenter la couverture du vocabulaire et améliorer la qualité de la phonémisation Ghomala'.
5.  **Évaluation des Modèles**: Mettre en place et exécuter les procédures d'évaluation objective (MCD) et subjective (MOS) tel que décrit dans `Experimental_Plan_and_Discussion.md` pour comparer les performances des modèles entraînés.
6.  **Optimisation des Performances**: Ajuster les hyperparamètres (y compris `segment_size` et `fp16_run: true`) pour optimiser l'utilisation du GPU et les performances.
7.  **Fine-tuning et Optimisation**: Après l'analyse comparative, ajuster les hyperparamètres et potentiellement fine-tuner les modèles pour améliorer les résultats.
8.  **Synthèse Vocale**: Utiliser les meilleurs modèles entraînés pour générer des échantillons audio à partir de textes spécifiques dans chaque langue.

Ce rapport sera mis à jour au fur et à mesure de l'avancement du projet.
