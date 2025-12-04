# Explication Détaillée des Paramètres de Configuration VITS

Ce document explique les différents paramètres trouvés dans les fichiers `config.json` du projet VITS Multilingue (`multilingual.json`, `french_only.json`, etc.). Comprendre ces paramètres est crucial pour configurer, entraîner et optimiser efficacement le modèle.

Chaque section correspond à une partie du fichier JSON de configuration.

## 1. Section `train` (Paramètres d'Entraînement)

Ces paramètres contrôlent le processus d'entraînement du modèle.

*   `log_interval` (Type: Entier):
    *   **Description**: Fréquence (en nombre d'itérations ou de pas d'entraînement) à laquelle les métriques d'entraînement (pertes, etc.) sont enregistrées et affichées dans la console/logs.
    *   **Importance**: Aide à surveiller la progression de l'entraînement en temps réel. Une valeur trop faible peut surcharger les logs, une valeur trop élevée peut rendre le suivi difficile.

*   `eval_interval` (Type: Entier):
    *   **Description**: Fréquence (en nombre d'itérations) à laquelle le modèle est évalué sur l'ensemble de validation et à laquelle les checkpoints (sauvegardes du modèle) sont enregistrés.
    *   **Importance**: Détermine la fréquence des sauvegardes du modèle et des évaluations de performance. Une valeur appropriée est essentielle pour la reprise d'entraînement et pour suivre la performance sans surcharger le système de fichiers.

*   `seed` (Type: Entier):
    *   **Description**: Graine numérique utilisée pour initialiser les générateurs de nombres pseudo-aléatoires.
    *   **Importance**: Assure la reproductibilité de l'entraînement. En utilisant la même graine, vous obtiendrez les mêmes résultats si vous relancez l'entraînement avec les mêmes données et paramètres.

*   `epochs` (Type: Entier):
    *   **Description**: Nombre total de fois que l'ensemble de données d'entraînement sera parcouru par le modèle.
    *   **Importance**: Détermine la durée maximale de l'entraînement. Un nombre trop faible peut entraîner un sous-apprentissage (le modèle n'apprend pas assez), un nombre trop élevé peut entraîner un surapprentissage (le modèle mémorise les données d'entraînement mais généralise mal).

*   `learning_rate` (Type: Virgule flottante):
    *   **Description**: Taille du pas utilisé par l'optimiseur pour ajuster les poids du modèle à chaque itération.
    *   **Importance**: L'un des hyperparamètres les plus critiques. Une `learning_rate` trop élevée peut empêcher la convergence (le modèle "saute" la solution optimale), une valeur trop faible peut rendre l'entraînement très lent.

*   `betas` (Type: Liste de virgules flottantes), `eps` (Type: Virgule flottante):
    *   **Description**: Paramètres spécifiques de l'optimiseur AdamW (souvent `[0.8, 0.99]` pour `betas` et `1e-9` pour `eps`).
    *   **Importance**: Affectent le comportement de l'optimiseur. Généralement, les valeurs par défaut fonctionnent bien.

*   `batch_size` (Type: Entier):
    *   **Description**: Nombre d'échantillons de données traités simultanément à chaque itération d'entraînement.
    *   **Importance**: Impacte directement la stabilité de l'entraînement et l'utilisation de la mémoire. Une `batch_size` plus grande peut accélérer l'entraînement sur GPU mais consomme plus de mémoire. Une `batch_size` de `1` (utilisée ici) minimise la consommation mémoire mais peut rendre l'entraînement plus lent et plus instable.

*   `fp16_run` (Type: Booléen):
    *   **Description**: Si `true`, active l'entraînement en précision mixte (Floating Point 16).
    *   **Importance**: Réduit la consommation de mémoire GPU (environ de moitié) et peut accélérer l'entraînement sur les GPUs modernes compatibles avec les Tensor Cores. Peut parfois introduire des problèmes de stabilité numérique. **Recommandé sur GPU.**

*   `lr_decay` (Type: Virgule flottante):
    *   **Description**: Taux de décroissance du taux d'apprentissage. Le taux d'apprentissage est multiplié par cette valeur à des intervalles spécifiques (souvent par époque).
    *   **Importance**: Aide le modèle à converger plus finement vers la fin de l'entraînement.

*   `segment_size` (Type: Entier):
    *   **Description**: Longueur (en échantillons audio) des segments audio extraits des fichiers WAV pour l'entraînement.
    *   **Importance**: Très important pour la consommation mémoire. Des segments plus courts consomment moins de mémoire mais peuvent potentiellement capturer moins de contexte prosodique. Des segments plus longs sont plus gourmands en mémoire. **Réduire cette valeur (ex: `4096`, `2048`) est une solution clé aux problèmes de mémoire.**

*   `init_lr_ratio` (Type: Virgule flottante), `warmup_epochs` (Type: Entier):
    *   **Description**: Paramètres pour un programme de "warmup" du taux d'apprentissage, où le taux d'apprentissage augmente progressivement au début de l'entraînement avant de suivre le programme principal.
    *   **Importance**: Peut améliorer la stabilité de l'entraînement, surtout au début, en évitant des ajustements trop agressifs des poids du modèle.

*   `c_mel` (Type: Virgule flottante), `c_kl` (Type: Virgule flottante):
    *   **Description**: Coefficients des différentes composantes de la fonction de perte du modèle (perte Mel-spectrogramme, perte KL-divergence).
    *   **Importance**: Équilibrer l'importance de chaque perte pour guider l'apprentissage du modèle.

## 2. Section `data` (Paramètres des Données)

Ces paramètres définissent comment les données d'entraînement et de validation sont chargées et prétraitées.

*   `training_files`, `validation_files` (Type: Chaîne de caractères):
    *   **Description**: Chemins vers les fichiers "filelist" qui contiennent la liste des chemins audio, IDs de locuteur et séquences de phonèmes pour l'entraînement et la validation.
    *   **Importance**: Indiquent au modèle où trouver les données à utiliser.

*   `text_cleaners` (Type: Liste de chaînes de caractères):
    *   **Description**: Liste des noms des fonctions de nettoyage de texte (phonémisation) à appliquer.
    *   **Importance**: Convertit le texte brut en une séquence normalisée d'unités compréhensibles par le modèle (phonèmes). **Crucial pour la qualité linguistique.** Pour le multilingue, `["multilingual_cleaners_v2"]` est utilisé. Pour monolingue, des nettoyeurs spécifiques comme `["french_cleaners"]` sont utilisés.

*   `max_wav_value` (Type: Virgule flottante):
    *   **Description**: Valeur maximale possible pour les échantillons audio (ex: `32768.0` pour audio 16-bit).
    *   **Importance**: Utilisé pour normaliser les échantillons audio.

*   `sampling_rate` (Type: Entier):
    *   **Description**: Fréquence d'échantillonnage de l'audio en Hz (ex: `22050`).
    *   **Importance**: **Critique**. Tous les fichiers audio du dataset doivent avoir cette fréquence d'échantillonnage. Le modèle est entraîné et génère de l'audio à cette fréquence.

*   `filter_length`, `hop_length`, `win_length` (Type: Entiers):
    *   **Description**: Paramètres pour la transformation de Fourier à court terme (STFT) qui convertit l'audio en spectrogrammes (représentations fréquentielle).
    *   **Importance**: Affectent la résolution temporelle et fréquentielle des spectrogrammes. Doivent être cohérents avec la façon dont les données sont préparées.

*   `n_mel_channels` (Type: Entier):
    *   **Description**: Nombre de bandes de filtre Mel dans les spectrogrammes Mel.
    *   **Importance**: Détermine la granularité de la représentation fréquentielle utilisée.

*   `mel_fmin`, `mel_fmax` (Type: Virgule flottante ou `null`):
    *   **Description**: Gamme de fréquences minimale et maximale pour la création des spectrogrammes Mel.
    *   **Importance**: Définit la plage de fréquences pertinentes pour la perception auditive humaine.

*   `add_blank` (Type: Booléen):
    *   **Description**: Si `true`, un symbole "blanc" est ajouté entre les phonèmes et au début/fin de la séquence de phonèmes.
    *   **Importance**: Utilisé pour faciliter l'alignement entre les phonèmes et l'audio, notamment dans les modèles basés sur l'architecture CTC.

*   `n_speakers` (Type: Entier):
    *   **Description**: Nombre total de locuteurs ou d'IDs de langue distincts dans l'ensemble de données.
    *   **Importance**: **Crucial pour l'entraînement multilingue**. Indique au modèle le nombre d'embeddings de locuteurs à apprendre. Si `n_speakers` est `1` (pour monolingue), cela signifie qu'il n'y a pas de conditionnement explicite par `speaker_id` (l'embedding de locuteur n'est pas créé).

*   `cleaned_text` (Type: Booléen):
    *   **Description**: Si `true`, indique que le texte dans les filelists a déjà été nettoyé/phonémisé.
    *   **Importance**: Permet au script d'entraînement de sauter l'étape de phonémisation si elle a déjà été effectuée lors de la préparation des données.

*   `n_symbols` (Type: Entier):
    *   **Description**: Nombre total de symboles uniques (phonèmes + symboles spéciaux comme le pad, le blank, la ponctuation) définis dans le jeu de symboles du modèle.
    *   **Importance**: Définit la taille de l'espace d'entrée pour l'encodeur de texte du modèle. Doit correspondre exactement au nombre de symboles dans `symbols_multilingual_v2.py` ou le fichier de symboles utilisé.

## 3. Section `model` (Paramètres du Modèle)

Ces paramètres définissent l'architecture interne et la capacité du modèle VITS.

*   `inter_channels`, `hidden_channels`, `filter_channels` (Type: Entiers):
    *   **Description**: Dimensions internes des canaux et des couches dans différentes parties du réseau (encodeurs, flow, etc.).
    *   **Importance**: Déterminent la "largeur" et la capacité du modèle à apprendre des représentations complexes. Des valeurs plus grandes augmentent la capacité (et potentiellement la qualité) mais aussi la consommation de mémoire et le temps de calcul.

*   `n_heads` (Type: Entier), `n_layers` (Type: Entier), `kernel_size` (Type: Entier), `p_dropout` (Type: Virgule flottante):
    *   **Description**: Paramètres des blocs Transformer utilisés dans les encodeurs du modèle (nombre de têtes d'attention, nombre de couches, taille du noyau de convolution, taux de dropout).
    *   **Importance**: Affectent la capacité du modèle à capturer les dépendances à long terme dans les séquences et à prévenir le surapprentissage.

*   `resblock`, `resblock_kernel_sizes`, `resblock_dilation_sizes` (Type: Chaîne de caractères, Listes d'entiers):
    *   **Description**: Paramètres des blocs résiduels utilisés dans le générateur de waveform.
    *   **Importance**: Contribuent à la capacité du générateur à créer des waveforms audio de haute qualité.

*   `upsample_rates`, `upsample_initial_channel`, `upsample_kernel_sizes` (Type: Listes d'entiers, Entier):
    *   **Description**: Paramètres pour les couches d'upsampling (déconvolution) dans le générateur, qui transforment les caractéristiques de bas niveau en une waveform audio de haute fréquence d'échantillonnage.
    *   **Importance**: Déterminent la qualité et la résolution de la waveform générée.

*   `n_layers_q` (Type: Entier):
    *   **Description**: Nombre de couches dans l'encodeur postérieur (qui encode l'audio en une représentation latente).
    *   **Importance**: Affecte la capacité de l'encodeur à extraire des représentations latentes pertinentes de l'audio.

*   `use_spectral_norm` (Type: Booléen):
    *   **Description**: Si `true`, applique la normalisation spectrale au discriminateur.
    *   **Importance**: Peut aider à stabiliser l'entraînement des réseaux adversariaux (GANs) en contrôlant la force du discriminateur.

*   `gin_channels` (Type: Entier):
    *   **Description**: Nombre de canaux pour le conditionnement global (global conditioning input), utilisé pour les embeddings de locuteur/langue.
    *   **Importance**: **Crucial pour les modèles multilingues**. Si `n_speakers > 1`, `gin_channels` doit être une valeur positive (souvent `192` ou `hidden_channels`) pour que les embeddings de locuteur soient créés et utilisés. Si `n_speakers = 1` (monolingue), il doit être `0` car le conditionnement global n'est pas utilisé.
