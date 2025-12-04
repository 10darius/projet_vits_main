# Rapport d'Activité du Projet VITS Multilingue (27 Novembre 2025)

Ce rapport documente les activités menées pour adapter le projet `vits_multilingual-main` à la génération de parole multilingue (Français, Ghomala', Anglais), en se concentrant sur la préparation des données, l'intégration des symboles phonétiques et la documentation.

## 1. Objectifs Initiaux du Projet

L'objectif principal était d'adapter le modèle VITS pour supporter le Français, le Ghomala' et l'Anglais, avec une considération future pour le Medumba. Cela impliquait la gestion des spécificités linguistiques de chaque langue, la création d'un jeu de symboles unifié, la préparation des données et la mise en place d'un pipeline de prétraitement robuste.

## 2. Problèmes Rencontrés et Solutions Implémentées

Au cours du projet, plusieurs défis ont été relevés et des solutions spécifiques ont été mises en œuvre :

### 2.1. Installation et Configuration de l'Environnement

*   **Problème :** Assurer la compatibilité des dépendances et la configuration correcte de l'environnement Conda (`mon_env39`).
*   **Solution :** Vérification et installation des paquets nécessaires, y compris `phonemizer` et sa dépendance `espeak-ng`. Le chemin d'accès à `libespeak-ng.dll` a été défini programmatiquement dans `tacotron-multilingual/text/cleaners.py` pour éviter les erreurs de localisation.

### 2.2. Gestion des Fichiers de Données et des Fréquences d'Échantillonnage

*   **Problème :** Incohérence des fréquences d'échantillonnage entre les datasets Français (44100 Hz) et Ghomala' (22050 Hz), et la fréquence d'échantillonnage par défaut du modèle.
*   **Solution :** Initialement, la fréquence d'échantillonnage du modèle (`hparams.sampling_rate`) a été ajustée à 44100 Hz pour le dataset Français. Cependant, cela a causé des problèmes avec le Ghomala'. La stratégie a été révisée pour **rééchantillonner les fichiers audio Français de 44100 Hz à 22050 Hz** pour assurer une cohérence sur l'ensemble des datasets.
*   **Problème :** Négliger les parties `part1` et `part2` du dataset français.
*   **Solution :** Modification du script `create_filelists.py` pour exclure ces parties.
*   **Problème :** Chemin des fichiers d'entraînement et de validation dans `hparams.py`.
*   **Solution :** Mise à jour des chemins vers des chemins absolus pour `training_files` et `validation_files`.

### 2.3. Traitement Audio (Mono/Stéréo)

*   **Problème :** Les fichiers audio stéréo provoquaient des erreurs lors du chargement.
*   **Solution :** Modification de la fonction `load_wav_to_torch` dans `tacotron-multilingual/utils.py` pour convertir automatiquement l'audio stéréo en mono.

### 2.4. Erreurs de Script et Débogage

*   **Problème :** `SyntaxError` dû à des barres obliques inverses non échappées dans les chemins de fichiers Windows (`preprocess_multilingual_data_v2.py`).
*   **Solution :** Correction des chaînes de chemin en utilisant des barres obliques normales ou des chaînes brutes (r"...") et en ajoutant `sys.path.append('.')` pour résoudre les `ModuleNotFoundError`.
*   **Problème :** `ValueError` lié aux fréquences d'échantillonnage, signalé lorsque les données prétraitées ne correspondaient pas à la configuration du modèle.
*   **Solution :** Mise à jour du message d'erreur avec une f-string pour un débogage plus facile et réalignement des fréquences d'échantillonnage des données.

### 2.5. Jeu de Symboles Unifié et Grapheme-to-Phoneme (G2P)

*   **Problème :** `NameError` dans `vits_multilingual-main/text/symbols.py` en raison d'un bloc de symboles dupliqué et incorrect.
*   **Solution :** Suppression du bloc de code en double.
*   **Problème :** Incohérence entre les phonèmes générés par `espeak` (pour l'anglais) et le jeu de symboles défini, notamment concernant la gestion du stress et de la ponctuation.
*   **Solution :** Affinement du script `discover_english_phonemes_v2.py` et ajustement de la fonction `english_g2p_v2` dans `vits_multilingual-main/text/cleaners_multilingual_v2.py` pour utiliser `with_stress=True` et `preserve_punctuation=False` dans `phonemize`, assurant ainsi une correspondance exacte.
*   **Problème :** Intégration des spécificités du Ghomala' (tons, coup de glotte, voyelles nasales, caractères accentués) dans le jeu de symboles.
*   **Solution :** Création de `vits_multilingual-main/text/symbols_multilingual_v2.py` incluant le coup de glotte (`ʔ`), l'apostrophe (`'`), l'alphabet latin complet, les voyelles nasales et accentuées du Ghomala', et les symboles de tons.

### 2.6. Lexique Ghomala'

*   **Problème :** Difficulté à extraire un lexique Ghomala' propre à partir du PDF du dictionnaire à cause de la qualité de l'OCR et de la complexité de la transcription IPA.
*   **Solution :** Extraction du texte brut du `DICTIONNAIRE_GHOMALA.pdf` via `pdftotext`. Il a été décidé d'utiliser l'orthographe du dictionnaire Ghomala' (avec ses diacritiques) comme représentation "phonétique" directe dans le lexique, car elle intègre déjà les informations tonales et les sons spécifiques. Un fichier `ghomala_lexicon.tsv` a été généré à partir des données structurées fournies, utilisant la colonne 'word' pour la représentation phonétique.

### 2.7. Documentation et Rapports

*   **Activité :** Rédaction de rapports et de documentations clés pour structurer le projet et les décisions.
*   **Livraisons :**
    *   `VITS_Adaptation_Report.md` : Rapport initial sur la stratégie d'adaptation de VITS.
    *   `Methodology_G2P_Lexicon.md` : Guide détaillé pour le G2P et la création de lexiques.
    *   `Experimental_Plan_and_Discussion.md` : Plan pour les expériences comparatives.
    *   `Linguistic_Deep_Dive_Ghomala.md` : Analyse linguistique approfondie du Ghomala', mise à jour avec les informations du dictionnaire.
    *   `Phoneme_Report_Multilingual.md` : Rapport détaillé sur les phonèmes multilingues, mis à jour avec les spécificités du Ghomala'.
    *   `create_lexicon_template.py` : Script utilitaire pour générer un template de lexique Ghomala'.
    *   `discover_english_phonemes_v2.py` : Script pour découvrir les phonèmes anglais via `espeak`.
    *   `preprocess_multilingual_data_v2.py` : Script principal pour le prétraitement des données.
    *   `filelists/train_processed.txt` et `filelists/val_processed.txt` : Listes de fichiers d'entraînement et de validation générées après prétraitement.
    *   `ghomala_lexicon.tsv`: Lexique Ghomala' généré à partir du CSV fourni par l'utilisateur.

## 3. Étapes Suivantes

Les prochaines étapes consisteront à :
1.  Générer le `README_PROJECT.md` pour un guide utilisateur complet.
2.  Fournir un résumé final des travaux réalisés et des instructions claires pour l'entraînement du modèle.
3.  Lancer l'entraînement du modèle VITS avec les données préparées.
4.  Évaluer le modèle et potentiellement affiner le lexique Ghomala' si nécessaire.