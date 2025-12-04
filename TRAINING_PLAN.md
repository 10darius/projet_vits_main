# Plan d'Entraînement Comparatif du Modèle VITS Multilingue

Ce document détaille un plan d'entraînement structuré pour le modèle VITS multilingue, visant une analyse comparative des performances en mode monolingue et multilingue.

## 1. Objectif

L'objectif est de :
1.  Entraîner le modèle sur chaque langue (Français, Anglais, Ghomala') individuellement pour établir des performances de référence.
2.  Entraîner le modèle sur l'ensemble des trois langues simultanément pour évaluer les bénéfices et les défis de l'approche multilingue (notamment le transfert de connaissances).
3.  Permettre une analyse comparative objective de la qualité de la synthèse vocale pour chaque configuration.

## 2. Configuration Générale

*   **Environnement**: L'environnement Conda `mon_env39` (Python 3.8.20) doit être activé (`conda activate mon_env39`) ou les commandes doivent être exécutées avec `conda run -n mon_env39 python ...`.
*   **Scripts**:
    *   `preprocess_multilingual_data_v2.py`: Pour la préparation des données.
    *   `vits_multilingual-main/train.py`: Pour l'entraînement du modèle.
*   **Configuration du modèle**: Le fichier `vits_multilingual-main/configs/multilingual.json` est le fichier de configuration principal. Des versions modifiées de ce fichier seront utilisées pour les entraînements monolingues.
*   **Sauvegarde des modèles**: Les checkpoints seront sauvegardés toutes les 10 époques (`eval_interval` dans la section `train` du config sera ajusté si nécessaire pour correspondre à cette fréquence ou les checkpoints seront manuellement gérés). Pour cet entraînement initial et comparatif, l'objectif est d'obtenir des modèles à des intervalles réguliers.

## 3. Déroulement de l'Entraînement

Chaque phase d'entraînement sera exécutée indépendamment. Il est recommandé d'utiliser un répertoire de logs distinct (`--model` argument) pour chaque expérimentation afin de conserver les checkpoints et les logs séparés.

### Phase 1 : Entraînement Monolingue

Pour chaque langue, nous allons créer un fichier de configuration spécifique et préparer des listes de fichiers d'entraînement et de validation ne contenant que les données de cette langue.

**Exemple de structure pour les fichiers de configuration monolingues (à créer dans `vits_multilingual-main/configs/`)**:

*   `french_only.json` (pour le Français)
*   `english_only.json` (pour l'Anglais)
*   `ghomala_only.json` (pour le Ghomala')

Ces fichiers devront adapter `multilingual.json` en :
*   `data.training_files` et `data.validation_files`: Pointer vers des filelists spécifiques à la langue.
*   `data.text_cleaners`: Utiliser le cleaner spécifique à la langue (e.g., `["french_cleaners"]`, `["english_cleaners2"]`, `["ghomala_cleaners1"]` - à vérifier dans les configs d'origine comme `french.json`, `bbj.json`, `me_base.json`).
*   `data.n_speakers`: `1` (ou `0` si le modèle VITS considère `0` pour le cas monolingue sans speaker embedding explicite, ou si l'on souhaite un embedding de speaker par défaut).
*   `data.n_symbols`: Sera ajusté pour n'inclure que les symboles pertinents à la langue.

**Préparation des Filelists Monolingues:**

Des scripts ou des modifications de `preprocess_multilingual_data_v2.py` seront nécessaires pour générer `filelists/train_fr.txt`, `filelists/val_fr.txt`, `filelists/train_en.txt`, etc.

**Commandes d'entraînement (exemple, à adapter)**:

```bash
# Français uniquement
conda run -n mon_env39 python vits_multilingual-main/train.py \
    --config vits_multilingual-main/configs/french_only.json \
    --model vits_monolingual_fr_10_epochs_save

# Anglais uniquement
conda run -n mon_env39 python vits_multilingual-main/train.py \
    --config vits_multilingual-main/configs/english_only.json \
    --model vits_monolingual_en_10_epochs_save

# Ghomala uniquement
conda run -n mon_env39 python vits_multilingual-main/train.py \
    --config vits_multilingual-main/configs/ghomala_only.json \
    --model vits_monolingual_gh_10_epochs_save
```

### Phase 2 : Entraînement Multilingue (3 Langues)

Cette phase utilisera la configuration `multilingual.json` déjà créée et les filelists `train_processed.txt` et `val_processed.txt` générées précédemment.

**Commande d'entraînement**:

```bash
conda run -n mon_env39 python vits_multilingual-main/train.py \
    --config vits_multilingual-main/configs/multilingual.json \
    --model vits_multilingual_fr_gh_en_10_epochs_save
```

## 4. Sauvegarde des Modèles (Checkpoints)

Chaque exécution d'entraînement sauvegardera des checkpoints régulièrement. Pour une analyse comparative après 10 époques initiales, il faudra surveiller les logs et les fichiers de checkpoint dans le répertoire `logs/<model_name>`.

## 5. Analyse Comparative

Une fois les entraînements terminés, une analyse comparative pourra être effectuée en évaluant la qualité de la synthèse vocale de chaque modèle (monolingue et multilingue) sur leurs langues respectives. Les métriques objectives (MCD) et subjectives (MOS) devront être utilisées comme détaillé dans `Experimental_Plan_and_Discussion.md`.

## 6. Prochaines Étapes

*   Créer les fichiers `french_only.json`, `english_only.json`, `ghomala_only.json` et leurs filelists correspondantes.
*   Lancer chaque entraînement en surveillant les ressources (CPU/GPU) et les logs.
*   Collecter les checkpoints pour la comparaison.
*   Procéder à l'analyse et à l'évaluation.
