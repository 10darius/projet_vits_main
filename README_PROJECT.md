# Projet VITS Multilingue : Synthèse Vocale avec Français, Ghomala' et Anglais

Ce projet vise à adapter le modèle VITS (Variational Inference with adversarial learning for Text-to-Speech) pour la synthèse vocale multilingue, en se concentrant initialement sur le Français, le Ghomala' et l'Anglais. Il inclut un pipeline complet pour la préparation des données, l'unification des symboles phonétiques et la génération de lexiques spécifiques.

## 1. Structure du Projet

Le projet est basé sur une version modifiée de `vits_multilingual-main` et inclut les répertoires et fichiers clés suivants :

*   `vits_multilingual-main/` : Répertoire principal du modèle VITS adapté.
    *   `text/symbols_multilingual_v2.py` : Jeu de symboles phonétiques unifié pour toutes les langues.
    *   `text/cleaners_multilingual_v2.py` : Fonctions de nettoyage et de conversion G2P multilingues.
*   `filelists/` : Contient les listes de fichiers générées pour l'entraînement et la validation (`train_processed.txt`, `val_processed.txt`).
*   `ghomala_lexicon.tsv` : Lexique Ghomala' généré, utilisant l'orthographe du dictionnaire comme représentation phonétique.
*   `preprocess_multilingual_data_v2.py` : Script maître pour le prétraitement de toutes les données.
*   `discover_english_phonemes_v2.py` : Utilitaire pour extraire les phonèmes anglais de `espeak`.
*   `create_lexicon_template.py` : Script pour générer un template de lexique Ghomala' (moins utilisé après l'intégration du CSV).
*   `resample_french_audio.py`: Script pour le rééchantillonnage des données audio françaises.
*   `Phoneme_Report_Multilingual.md` : Rapport détaillé sur les phonèmes de chaque langue.
*   `Linguistic_Deep_Dive_Ghomala.md` : Analyse linguistique du Ghomala'.
*   `Activity_Report.md` : Journal des activités et des solutions implémentées.
*   `GEMINI.md`: Fichier de mémoire du projet.

## 2. Configuration et Prérequis

### 2.1. Environnement Conda

Il est fortement recommandé d'utiliser un environnement Conda pour gérer les dépendances. L'environnement `mon_env39` est utilisé pour ce projet.

```bash
conda create -n mon_env39 python=3.9
conda activate mon_env39
```

### 2.2. Installation des Dépendances Python

Installez les bibliothèques Python requises. Un fichier `requirements.txt` devrait être disponible ou peut être généré à partir des imports dans les scripts. Les dépendances critiques incluent `phonemizer`, `torch`, `torchaudio`, `numpy`, `scipy`, `unidecode`, `inflect`, `librosa`, etc.

```bash
pip install -r requirements.txt # Si disponible
# Ou installer manuellement les paquets essentiels :
pip install phonemizer torch torchvision torchaudio numpy scipy unidecode inflect librosa
```

### 2.3. Installation de eSpeak-NG

`espeak-ng` est nécessaire pour la phonémisation de l'anglais et du français.

*   Téléchargez et installez `espeak-ng` (version 1.52.0 ou ultérieure) depuis [espeak-ng GitHub](https://github.com/espeak-ng/espeak-ng/releases).
*   Assurez-vous que le chemin vers `libespeak-ng.dll` (sous Windows) ou `libespeak-ng.so` (sous Linux) est correctement configuré dans `vits_multilingual-main/text/cleaners.py` ou via une variable d'environnement si nécessaire. Le projet utilise la détection automatique ou le chemin `C:\Program Files\eSpeak NG\libespeak-ng.dll` sous Windows.

## 3. Préparation des Données

Le pipeline de prétraitement est essentiel pour préparer les données audio et texte pour l'entraînement du modèle VITS.

### 3.1. Organisation des Datasets

Vos données audio et texte pour chaque langue doivent être organisées. Les scripts de prétraitement s'attendent à ce que les fichiers soient structurés de manière cohérente.

*   **Français :** Les fichiers `train.txt` et `val.txt` sont générés en ignorant `part1` et `part2` du dataset. La fréquence d'échantillonnage de 22050 Hz est supposée.
*   **Ghomala' :** Un lexique (`ghomala_lexicon.tsv`) est utilisé, où les "phonèmes" sont les symboles orthographiques accentués.
*   **Anglais :** La phonémisation est effectuée par `phonemizer` (backend `espeak`).

### 3.2. Prétraitement Multilingue

Le script `preprocess_multilingual_data_v2.py` est le point d'entrée pour le prétraitement de toutes les langues. Il générera les listes de fichiers `train_processed.txt` et `val_processed.txt` dans le répertoire `filelists/`. Ce script intègre une vérification robuste de l'existence des fichiers audio pour éviter les erreurs.

```bash
conda run -n mon_env39 python preprocess_multilingual_data_v2.py
```

**Note :** Ce script gère :
*   La conversion audio stéréo vers mono.
*   L'application des nettoyeurs de texte spécifiques à chaque langue.
*   L'intégration du lexique Ghomala'.
*   La génération des paires `(chemin_audio|texte_nettoyé|locuteur_id)`.

## 4. Prochaines Étapes : Entraînement du Modèle VITS

Une fois les données prétraitées et les fichiers `train_processed.txt` et `val_processed.txt` générés, vous pouvez passer à l'entraînement du modèle VITS.

1.  **Vérifiez les configurations** : Assurez-vous que les fichiers de configuration du modèle VITS (par exemple, dans `vits_multilingual-main/configs/`) sont correctement adaptés pour le multilingue, notamment en ce qui concerne le nombre de locuteurs (speaker IDs) et le jeu de symboles.
    *   **Speaker IDs :** Français (commence à 0), Ghomala' (commence à 10), Anglais (commence à 20).
2.  **Lancez l'entraînement** : Exécutez le script d'entraînement principal du modèle VITS.

```bash
# Commande d'entraînement
conda run -n mon_env39 python vits_multilingual-main/train.py \
    --config vits_multilingual-main/configs/multilingual.json \
    --model vits_multilingual_fr_gh_en
```

## 5. Synthèse et Évaluation

Après l'entraînement, vous pouvez utiliser le modèle pour synthétiser de la parole à partir de texte et évaluer ses performances.

1.  **Synthèse :** Utilisez le script d'inférence (souvent un notebook IPython `inference.ipynb` ou un script Python dédié) pour générer des échantillons audio.
2.  **Évaluation :** Évaluez la qualité de la parole synthétisée pour chaque langue, en portant une attention particulière à la prononciation, à la fluidité et, pour le Ghomala', au rendu des tons.

---

**Contributions et Maintenance :**

Ce projet a bénéficié de contributions pour résoudre les problèmes de prétraitement, d'intégration linguistique et de structuration du pipeline. Pour toute question ou amélioration, veuillez consulter les rapports détaillés et les scripts fournis.