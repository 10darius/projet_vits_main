# Guide Complet : Projet VITS Multilingue (Préparation, Entraînement et Inférence)

Ce document sert de guide détaillé pour la mise en place, la préparation des données, l'entraînement et l'inférence avec le modèle VITS (Variational Inference with adversarial learning for Text-to-Speech) dans un contexte multilingue (Français, Anglais, Ghomala'). Il inclut des commandes spécifiques et des explications des processus sous diverses perspectives.

## Table des Matières
1.  Introduction au Projet
2.  Prérequis et Configuration de l'Environnement
3.  Concepts Clés du Modèle VITS
4.  Organisation des Données
5.  Préparation des Données (Preprocessing)
    *   5.1. Préparation des Données Monolingues
    *   5.2. Préparation des Données Multilingues Globales
6.  Fichiers de Configuration du Modèle
7.  Entraînement du Modèle
    *   7.1. Procédure Générale d'Entraînement
    *   7.2. Entraînement Monolingue (Français, Anglais, Ghomala')
    *   7.3. Entraînement Multilingue
8.  Inférence (Synthèse Vocale)
    *   8.1. Procédure Générale d'Inférence
    *   8.2. Inférence Monolingue
    *   8.3. Inférence Multilingue
9.  Analyse Comparative et Optimisation
10. Dépannage et Limitations Actuelles

---

## 1. Introduction au Projet

Le projet VITS Multilingue a pour but d'adapter le modèle VITS pour la synthèse vocale de plusieurs langues, notamment le Français, le Ghomala' et l'Anglais. L'objectif est de pouvoir entraîner des modèles spécialisés (monolingues) pour chaque langue ainsi qu'un modèle global (multilingue) capable de synthétiser la parole dans les trois langues. Une analyse comparative permettra d'évaluer les avantages de chaque approche.

## 2. Prérequis et Configuration de l'Environnement

### 2.1. Environnement Conda
Il est **impératif** d'utiliser l'environnement Conda `mon_env39` (Python 3.8.20) pour assurer la compatibilité des dépendances.

```bash
# Vérifier l'existence et activer l'environnement
conda env list
conda activate mon_env39
```
**Toutes les commandes `python` suivantes devront être précédées de `conda run -n mon_env39` pour garantir l'exécution dans le bon environnement.**

### 2.2. Installation des Dépendances Python
Assurez-vous que toutes les bibliothèques Python nécessaires sont installées. Un fichier `requirements.txt` devrait être présent à la racine du projet ou dans `vits_multilingual-main/`.

```bash
conda run -n mon_env39 pip install -r requirements.txt
# Ou, si vous rencontrez des problèmes, installez les paquets critiques manuellement:
conda run -n mon_env39 pip install phonemizer torch torchvision torchaudio numpy scipy unidecode inflect librosa tqdm
```

### 2.3. Installation et Configuration de eSpeak-NG
`espeak-ng` est utilisé par `phonemizer` pour la phonémisation.
*   Téléchargez et installez `espeak-ng` (version 1.52.0 ou ultérieure) depuis [espeak-ng GitHub](https://github.com/espeak-ng/espeak-ng/releases).
*   Le chemin vers `libespeak-ng.dll` (Windows) ou `libespeak-ng.so` (Linux) doit être accessible. Dans ce projet, le chemin est souvent configuré programmatiquement.

### 2.4. Configuration GPU (Essentiel pour l'entraînement)
Pour des performances et une gestion de la mémoire adéquates, l'utilisation d'un GPU est **indispensable**. Si votre environnement PyTorch n'indique pas "CUDA is available", vous devez :
*   Mettre à jour vos pilotes NVIDIA.
*   Installer le CUDA Toolkit et cuDNN compatibles.
*   Réinstaller PyTorch dans `mon_env39` avec le support CUDA (ex: `conda install pytorch torchvision torchaudio cudatoolkit=<version_cuda> -c pytorch`).

## 3. Concepts Clés du Modèle VITS

*   **Modèle VITS**: Un modèle de synthèse vocale "end-to-end" basé sur l'inférence variationnelle et l'apprentissage adversaire, capable de générer de la parole naturelle à partir de texte.
*   **Phonèmes**: Le modèle ne traite pas directement les lettres, mais des unités sonores (phonèmes). Ces phonèmes sont définis dans un **jeu de symboles unifié** (`vits_multilingual-main/text/symbols_multilingual_v2.py`).
*   **Lexique (G2P)**: Pour certaines langues (comme le Ghomala'), un lexique (`ghomala_lexicon.tsv`) est utilisé pour mapper les mots à leurs séquences de phonèmes. Pour d'autres langues (Français, Anglais), des "cleaners" (phonémiseurs) automatiques sont utilisés.
*   **Speaker ID (ID de Locuteur)**: Un identifiant numérique unique associé à chaque locuteur ou langue. Dans un modèle multilingue, il permet au modèle de distinguer la langue/voix à générer. (Ex: Français=0, Ghomala'=10, Anglais=20).
*   **Checkpoint**: Une sauvegarde de l'état du modèle (poids, optimiser, époque) à un instant T, permettant de reprendre l'entraînement ou de faire de l'inférence.

## 4. Organisation des Données

Vos données audio et texte pour chaque langue doivent être organisées de manière cohérente. Chaque dataset doit avoir des fichiers texte (`train.txt`, `val.txt`) listant les chemins des audios et leurs transcriptions, ainsi qu'un répertoire `wav/` (ou similaire) contenant les fichiers `.wav` correspondants. La fréquence d'échantillonnage de tous les audios doit être de **22050 Hz**.

Exemple de structure :
```
.
├── vits_multilingual-main/
│   ├── dataset_fr/
│   │   ├── train_fr.txt
│   │   ├── val_fr.txt
│   │   └── wav_fr/ (contenant les .wav français)
│   ├── dataset_bbj/ (pour Ghomala')
│   │   ├── train.txt
│   │   ├── test.txt (utilisé pour validation)
│   │   └── wav/ (contenant les .wav ghomala')
│   ├── dataset/ (pour Anglais)
│   │   ├── me_train.txt
│   │   ├── me_val.txt
│   │   └── wavs/ (contenant les .wav anglais)
├── filelists/ (contiendra les filelists générées)
├── ghomala_lexicon.tsv
├── preprocess_monolingual_data.py
├── preprocess_multilingual_data_v2.py
└── ... (autres fichiers projet)
```

## 5. Préparation des Données (Preprocessing)

Cette étape convertit vos données brutes en un format utilisable par le modèle VITS, en générant des "filelists" (listes de fichiers).

### 5.1. Préparation des Données Monolingues

Ce script génère des filelists séparées pour chaque langue.

*   **Commande**:
    ```bash
    conda run -n mon_env39 python preprocess_monolingual_data.py
    ```
*   **Processus (Détail)**:
    *   **Linguistiquement**: Pour chaque langue, le script lit les transcriptions textuelles et les convertit en séquences de phonèmes (IDs numériques) :
        *   **Français/Anglais**: Utilise `phonemizer` avec des nettoyeurs spécifiques (`french_cleaners`, `english_cleaners2`).
        *   **Ghomala'**: Effectue une recherche dans `ghomala_lexicon.tsv`. Si un mot est trouvé, sa séquence de symboles orthographiques accentués (représentant les phonèmes et les tons) est utilisée. Si non, un "fallback" basé sur les caractères est appliqué (voir section 10 sur les limitations).
    *   **Mathématiquement**: La conversion texte-phonèmes est une application de règles de substitution (pour les nettoyeurs) ou un mappage direct (pour le lexique). Chaque phonème est ensuite encodé en un ID numérique unique (`n_symbols`).
    *   **Informatiquement**: Le script itère sur chaque ligne des fichiers `train.txt`/`val.txt` de chaque langue. Il construit le chemin complet vers l'audio, vérifie son existence (`os.path.getsize`), génère la séquence de phonèmes, et formate la ligne comme `chemin_audio|speaker_id|id_phoneme_1 id_phoneme_2 ...`. Ces lignes sont écrites dans des fichiers de sortie spécifiques à chaque langue (ex: `filelists/train_fr.txt`, `filelists/val_gh.txt`).
    *   **Naïvement**: Imaginez que vous avez une grande pile de livres (vos audios et textes). Ce script feuillette chaque livre, prend le texte, le "traduit" en un code secret compréhensible par le modèle (les phonèmes), et note où trouver le fichier audio correspondant, ainsi que la "voix" à utiliser pour ce livre (le `speaker_id`). Il fait cela pour chaque langue séparément, créant une nouvelle pile de "notes" pour chaque langue.

### 5.2. Préparation des Données Multilingues Globales

Ce script génère un ensemble de filelists combinées pour toutes les langues.

*   **Commande**:
    ```bash
    conda run -n mon_env39 python preprocess_multilingual_data_v2.py
    ```
*   **Processus (Détail)**:
    *   Similaire à la préparation monolingue, mais toutes les lignes traitées de toutes les langues sont agrégées dans une seule liste, mélangées, puis écrites dans `filelists/train_processed.txt` et `filelists/val_processed.txt`. Chaque ligne inclut le `speaker_id` approprié pour distinguer la langue.
    *   **Naïvement**: Au lieu de faire des piles de notes séparées pour chaque langue, ce script mélange toutes les notes dans deux grandes piles : une pour l'entraînement (combinée) et une pour la validation (combinée). Chaque note contient toujours l'indication de la langue/voix (`speaker_id`).

## 6. Fichiers de Configuration du Modèle

Les configurations du modèle sont définies dans des fichiers JSON situés dans `vits_multilingual-main/configs/`.

*   **Monolingue**:
    *   `french_only.json` (n_speakers=1, n_symbols=145, gin_channels=0, text_cleaners=["french_cleaners"])
    *   `english_only.json` (n_speakers=1, n_symbols=526, gin_channels=0, text_cleaners=["english_cleaners2"])
    *   `ghomala_only.json` (n_speakers=1, n_symbols=110, gin_channels=0, text_cleaners=["ghomala_cleaners1"])
*   **Multilingue**:
    *   `multilingual.json` (n_speakers=21, n_symbols=624, gin_channels=192, text_cleaners=["multilingual_cleaners_v2"])

**Paramètres clés**:
*   `n_speakers`: Nombre total de locuteurs (ou IDs de langue) que le modèle doit apprendre. Pour monolingue: 1. Pour multilingue: le max(speaker_id)+1.
*   `n_symbols`: Nombre total de symboles (phonèmes) uniques dans le jeu de symboles unifié du modèle.
*   `gin_channels`: Dimension de l'embedding du locuteur. Set à 0 pour monolingue (car l'embedding n'est pas utilisé), et à 192 (ou `hidden_channels`) pour multilingue.
*   `text_cleaners`: Spécifie les nettoyeurs de texte à utiliser pour la phonémisation.

## 7. Entraînement du Modèle

### 7.1. Procédure Générale d'Entraînement

*   **Commande**: `conda run -n mon_env39 python vits_multilingual-main/train.py --config <chemin_config.json> --model <nom_modele>`
*   **Reprise d'entraînement**: Le script `train.py` est conçu pour reprendre automatiquement l'entraînement à partir du dernier checkpoint sauvegardé. Assurez-vous que le répertoire `logs/<nom_modele>` (où `<nom_modele>` est l'argument `--model` que vous avez fourni) existe et contient des fichiers `G_*.pth` et `D_*.pth`.
*   **Sauvegarde**: Les checkpoints sont sauvegardés régulièrement (défini par `eval_interval` dans le config).
*   **Processus (Détail)**:
    *   **Linguistiquement**: Le modèle apprend les relations entre les séquences de phonèmes (représentant la structure phonétique et tonale de la langue) et les caractéristiques acoustiques correspondantes. Pour le multilingue, il apprend également à distinguer et à générer ces caractéristiques en fonction de l'ID de locuteur fourni.
    *   **Mathématiquement**: Le modèle VITS est un Variational Autoencoder (VAE) conditionnel avec des composants adversaires. L'entraînement minimise plusieurs fonctions de perte (générateur, discriminateur, feature matching, KL divergence, duration loss) pour que le modèle génère de la parole réaliste et alignée avec le texte. La rétropropagation du gradient ajuste des millions de paramètres du réseau.
    *   **Informatiquement**: Le script charge les données par lots (`batch_size`), effectue une passe avant (forward pass) à travers le réseau, calcule les pertes, puis une passe arrière (backward pass) pour ajuster les poids du modèle. Ce cycle est répété pour un grand nombre d'époques. La charge de calcul est immense.
    *   **Naïvement**: Imaginez que le modèle est un élève qui apprend à parler. On lui donne des "notes" (filelists) avec des "codes secrets" (phonèmes) et des enregistrements de voix. Il essaie de reproduire ces voix en utilisant les codes. À chaque essai, il compare sa voix avec l'originale, trouve ses erreurs, et ajuste la façon dont il parle. Il fait cela des millions de fois, et de temps en temps, il fait une pause pour sauvegarder "ce qu'il a appris" (checkpoint). Pour le multilingue, on lui dit aussi "parle comme X" ou "parle comme Y" (`speaker_id`).

### 7.2. Entraînement Monolingue (10 époques pour comparaison)

Pour chaque langue, lancez l'entraînement comme suit, en remplaçant `<langue>` par `fr`, `en`, ou `gh`.

*   **Français uniquement**:
    ```bash
    conda run -n mon_env39 python vits_multilingual-main/train.py \
        --config vits_multilingual-main/configs/french_only.json \
        --model vits_monolingual_fr_10_epochs_save
    ```
*   **Anglais uniquement**:
    ```bash
    conda run -n mon_env39 python vits_multilingual-main/train.py \
        --config vits_multilingual-main/configs/english_only.json \
        --model vits_monolingual_en_10_epochs_save
    ```
*   **Ghomala' uniquement**:
    ```bash
    conda run -n mon_env39 python vits_multilingual-main/train.py \
        --config vits_multilingual-main/configs/ghomala_only.json \
        --model vits_monolingual_gh_10_epochs_save
    ```

### 7.3. Entraînement Multilingue (10 époques pour comparaison)

*   **3 Langues (Français + Anglais + Ghomala')**:
    ```bash
    conda run -n mon_env39 python vits_multilingual-main/train.py \
        --config vits_multilingual-main/configs/multilingual.json \
        --model vits_multilingual_fr_gh_en_10_epochs_save
    ```

## 8. Inférence (Synthèse Vocale)

L'inférence consiste à utiliser un modèle entraîné pour générer de la parole à partir d'un texte d'entrée.

### 8.1. Procédure Générale d'Inférence

*   **Utilisation**: Typiquement via `inference.py` ou un notebook `inference.ipynb`.
*   **Input**: Texte à synthétiser et un modèle entraîné (checkpoint). Pour le multilingue, un `speaker_id` est également requis.
*   **Output**: Fichier(s) audio `.wav` de la parole synthétisée.
*   **Processus (Détail)**:
    *   **Linguistiquement**: Le texte d'entrée est d'abord phonémisé (converti en séquence de phonèmes/IDs) en utilisant les nettoyeurs appropriés. Cette séquence représente la prononciation désirée.
    *   **Mathématiquement**: Le modèle effectue une passe avant (forward pass) pour transformer la séquence de phonèmes en caractéristiques acoustiques, puis utilise un décodeur pour transformer ces caractéristiques en waveform audio. Le `speaker_id` (pour le multilingue) conditionne le style et la prosodie de la voix générée.
    *   **Informatiquement**: Le script charge le modèle entraîné, prend le texte, le nettoie et le convertit en IDs. Il effectue l'inférence via le modèle PyTorch et sauvegarde l'audio résultant.
    *   **Naïvement**: Après avoir "appris à parler", l'élève (modèle) reçoit un nouveau texte et on lui dit "lis ce texte comme X". Il utilise tout ce qu'il a appris pour prononcer le texte, et le résultat est un enregistrement vocal.

### 8.2. Inférence Monolingue

Utilisez le modèle entraîné pour une seule langue. Vous n'aurez pas besoin de spécifier un `speaker_id` explicite si le modèle a été entraîné avec `n_speakers=1` (où l'ID est implicitement 0).

*   **Exemple de commande (à adapter selon le script d'inférence)**:
    ```bash
    conda run -n mon_env39 python vits_multilingual-main/inference.py \
        --model_path ./logs/vits_monolingual_fr_10_epochs_save/G_dernier_checkpoint.pth \
        --config_path vits_multilingual-main/configs/french_only.json \
        --text "Bonjour, ceci est un test de synthèse vocale." \
        --output_path "output_fr_mono.wav"
    ```

### 8.3. Inférence Multilingue

Utilisez le modèle multilingue entraîné, en spécifiant l'ID de locuteur pour la langue désirée.

*   **Exemple de commande (à adapter selon le script d'inférence)**:
    ```bash
    # Synthèse en Français avec le modèle multilingue
    conda run -n mon_env39 python vits_multilingual-main/inference.py \
        --model_path ./logs/vits_multilingual_fr_gh_en_10_epochs_save/G_dernier_checkpoint.pth \
        --config_path vits_multilingual-main/configs/multilingual.json \
        --text "Bonjour, je peux parler plusieurs langues." \
        --speaker_id 0 \
        --output_path "output_fr_multi.wav"

    # Synthèse en Anglais avec le modèle multilingue
    conda run -n mon_env39 python vits_multilingual-main/inference.py \
        --model_path ./logs/vits_multilingual_fr_gh_en_10_epochs_save/G_dernier_checkpoint.pth \
        --config_path vits_multilingual-main/configs/multilingual.json \
        --text "Hello, I can speak multiple languages." \
        --speaker_id 20 \
        --output_path "output_en_multi.wav"

    # Synthèse en Ghomala' avec le modèle multilingue
    conda run -n mon_env39 python vits_multilingual-main/inference.py \
        --model_path ./logs/vits_multilingual_fr_gh_en_10_epochs_save/G_dernier_checkpoint.pth \
        --config_path vits_multilingual-main/configs/multilingual.json \
        --text "Votre texte en Ghomala' ici." \
        --speaker_id 10 \
        --output_path "output_gh_multi.wav"
    ```

## 9. Analyse Comparative et Optimisation

Comparer les résultats d'inférence des modèles monolingues et multilingues permettra de déterminer les meilleures approches. Référez-vous à `Experimental_Plan_and_Discussion.md` pour les métriques d'évaluation (MCD, MOS).

## 10. Dépannage et Limitations Actuelles

Consultez le `SITUATION_REPORT.md` pour une vue d'ensemble des problèmes résolus, des limitations actuelles (notamment l'entraînement bloqué sur CPU sans GPU), et la feuille de route détaillée.

---
**Contributions et Maintenance :**

Ce projet a bénéficié de contributions pour résoudre les problèmes de prétraitement, d'intégration linguistique et de structuration du pipeline. Pour toute question ou amélioration, veuillez consulter les rapports détaillés et les scripts fournis.
