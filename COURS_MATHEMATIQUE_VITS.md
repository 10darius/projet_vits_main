# COURS MATHÉMATIQUE : VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)

## 📚 TABLE DES MATIÈRES
1. [Introduction](#1-introduction)
2. [Fondements Mathématiques](#2-fondements-mathématiques)
3. [Architecture du Modèle](#3-architecture-du-modèle)
4. [Formules Clés](#4-formules-clés)
5. [Algorithmes](#5-algorithmes)

---

## 1. INTRODUCTION

VITS est un modèle de synthèse vocale (Text-to-Speech) qui combine :
- **VAE (Variational Autoencoder)** : Modélisation probabiliste
- **GAN (Generative Adversarial Network)** : Apprentissage adversarial
- **Normalizing Flows** : Transformations inversibles

---

## 2. FONDEMENTS MATHÉMATIQUES

### 2.1 Probabilités et Distributions

#### Distribution Gaussienne (Normale)
```
p(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
```
- **μ** : moyenne
- **σ²** : variance
- **log(σ)** : log-variance (utilisé dans le code pour stabilité numérique)

#### Divergence de Kullback-Leibler (KL)
Mesure la différence entre deux distributions p et q :
```
KL(q||p) = ∫ q(x) log(q(x)/p(x)) dx
```

Pour deux gaussiennes :
```
KL(N(μ₁,σ₁²) || N(μ₂,σ²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
```

**Dans le code (losses.py, ligne 48-59)** :
```python
def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l
```

### 2.2 Variational Autoencoder (VAE)

#### Objectif ELBO (Evidence Lower Bound)
```
log p(x) ≥ E_q[log p(x|z)] - KL(q(z|x) || p(z))
         = ELBO
```

- **p(x|z)** : vraisemblance (likelihood)
- **q(z|x)** : encodeur (posterior approximation)
- **p(z)** : prior (généralement N(0,I))

#### Reparameterization Trick
Pour échantillonner z ~ N(μ, σ²) de manière différentiable :
```
z = μ + σ * ε,  où ε ~ N(0,1)
```

**Dans le code (models.py, ligne 234)** :
```python
z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
```

---

## 3. ARCHITECTURE DU MODÈLE

### 3.1 Encodeur de Texte (TextEncoder)

**Rôle** : Convertir phonèmes → représentation latente

#### Embedding
Transformation : indices discrets → vecteurs continus
```
e_i = W[i] ∈ ℝ^d
```
Normalisation : `e_i * √d` (comme dans Transformer)

**Code (models.py, ligne 145-147)** :
```python
self.emb = nn.Embedding(n_vocab, hidden_channels)
nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
x = self.emb(x) * math.sqrt(self.hidden_channels)
```

#### Projection vers μ et log(σ)
```
[μ, log(σ)] = Conv1D(h)
```
- **μ** : moyenne de la distribution latente
- **log(σ)** : log-écart-type (pour stabilité)

### 3.2 Encodeur Postérieur (PosteriorEncoder)

**Rôle** : Encoder le mel-spectrogramme en représentation latente

#### Formule
```
z ~ q(z|y) = N(μ_q, σ_q²)
μ_q, log(σ_q) = Encoder(y)
z = μ_q + σ_q * ε
```

**Code (models.py, ligne 232-235)** :
```python
stats = self.proj(x) * x_mask
m, logs = torch.split(stats, self.out_channels, dim=1)
z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
```

### 3.3 Normalizing Flow (ResidualCouplingBlock)

**Principe** : Transformation inversible pour augmenter l'expressivité

#### Coupling Layer (Couche de Couplage)
Divise l'entrée en deux parties [x_a, x_b] :
```
y_a = x_a
y_b = x_b * exp(s(x_a)) + t(x_a)
```
- **s(x_a)** : fonction de scale (échelle)
- **t(x_a)** : fonction de translation

#### Jacobien et Log-déterminant
Pour une transformation y = f(x), le changement de densité :
```
p_y(y) = p_x(x) / |det(∂f/∂x)|
log p_y(y) = log p_x(x) - log|det(∂f/∂x)|
```

Pour le coupling layer :
```
log|det(∂f/∂x)| = Σ s(x_a)
```

### 3.4 Prédicteur de Durée Stochastique (StochasticDurationPredictor)

**Objectif** : Prédire la durée de chaque phonème de manière probabiliste

#### Formulation
```
w ~ p(w|x) où w = durées
log p(w|x) = log p(z) - log|det(∂f/∂w)|
```

#### Transformation Log-Flow
```
z₀ = log(w)
z = Flow(z₀)
```

**Code (models.py, ligne 68-73)** :
```python
logdet_tot = 0
z0, logdet = self.log_flow(z0, x_mask)
logdet_tot += logdet
z = torch.cat([z0, z1], 1)
for flow in flows:
    z, logdet = flow(z, x_mask, g=x, reverse=reverse)
```

### 3.5 Monotonic Alignment Search (MAS)

**Problème** : Aligner texte et audio sans supervision

#### Formulation
Trouver l'alignement optimal A qui maximise :
```
A* = argmax_A Σ_t log p(y_t | x_{A(t)})
```

#### Negative Cross-Entropy
```
-H(y, x) = -Σ log p(y_t | x_s)
```

**Code (models.py, ligne 461-467)** :
```python
s_p_sq_r = torch.exp(-2 * logs_p)
neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)
neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
```

---

## 4. FORMULES CLÉS

### 4.1 Fonction de Perte Totale

```
L_total = L_recon + L_kl + L_dur + L_adv + L_fm
```

#### 4.1.1 Perte de Reconstruction (L_recon)
```
L_recon = ||y - ŷ||²
```
Mesure la différence entre audio réel et généré

#### 4.1.2 Perte KL (L_kl)
```
L_kl = KL(q(z|y) || p(z|x))
```
**Code (losses.py)** :
```python
kl = logs_p - logs_q - 0.5
kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
```

#### 4.1.3 Perte de Durée (L_dur)
Pour le prédicteur stochastique :
```
L_dur = -log p(w|x)
```

Pour le prédicteur déterministe :
```
L_dur = MSE(log(w), log(ŵ))
```

**Code (models.py, ligne 476-481)** :
```python
if self.use_sdp:
    l_length = self.dp(x, x_mask, w, g=g)
    l_length = l_length / torch.sum(x_mask)
else:
    logw_ = torch.log(w + 1e-6) * x_mask
    l_length = torch.sum((logw - logw_)**2, [1,2]) / torch.sum(x_mask)
```

#### 4.1.4 Perte Adversariale (L_adv)

**Discriminateur** :
```
L_D = E[(1 - D(y_real))²] + E[D(y_fake)²]
```

**Générateur** :
```
L_G = E[(1 - D(y_fake))²]
```

**Code (losses.py, ligne 18-30)** :
```python
def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
```

#### 4.1.5 Perte de Feature Matching (L_fm)
```
L_fm = Σ ||φ_i(y_real) - φ_i(y_fake)||₁
```
où φ_i sont les features intermédiaires du discriminateur

**Code (losses.py, ligne 7-15)** :
```python
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2
```

### 4.2 Rational Quadratic Spline (RQS)

Transformation non-linéaire pour les flows :

#### Forward
```
y = h(x) = y_k + (y_{k+1} - y_k) * [s_k * (x - x_k)² + d_k * (x - x_k) * (x_{k+1} - x)] / 
           [s_k * (x - x_k) + d_k * (x_{k+1} - x) + d_{k+1} * (x - x_k)]
```

#### Inverse
Résolution d'équation quadratique :
```
a * ξ² + b * ξ + c = 0
ξ = (2c) / (-b - √(b² - 4ac))
```

**Code (transforms.py, ligne 155-165)** :
```python
a = (((inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
      + input_heights * (input_delta - input_derivatives)))
b = (input_heights * input_derivatives
     - (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta))
c = - input_delta * (inputs - input_cumheights)
discriminant = b.pow(2) - 4 * a * c
root = (2 * c) / (-b - torch.sqrt(discriminant))
```

---

## 5. ALGORITHMES

### 5.1 Algorithme d'Entraînement

```
Pour chaque batch (x, y) :
    1. Encoder le texte : μ_p, σ_p = TextEncoder(x)
    2. Encoder l'audio : z, μ_q, σ_q = PosteriorEncoder(y)
    3. Flow : z_p = Flow(z)
    4. Alignement : A = MonotonicAlign(z_p, μ_p, σ_p)
    5. Durée : L_dur = DurationPredictor(x, A)
    6. Génération : ŷ = Generator(z)
    7. Discrimination : D_real, D_fake = Discriminator(y, ŷ)
    
    8. Calculer les pertes :
       L_kl = KL(q(z|y) || p(z|x))
       L_dur = -log p(w|x)
       L_adv = (1 - D_fake)²
       L_fm = ||φ(y) - φ(ŷ)||
    
    9. Backpropagation et mise à jour
```

### 5.2 Algorithme d'Inférence

```
Entrée : texte x
1. Encoder : μ_p, σ_p = TextEncoder(x)
2. Prédire durée : w = DurationPredictor(x)
3. Expansion : μ_p', σ_p' = Expand(μ_p, σ_p, w)
4. Échantillonner : z_p ~ N(μ_p', σ_p')
5. Flow inverse : z = Flow⁻¹(z_p)
6. Générer : y = Generator(z)
Sortie : audio y
```

---

## 6. CONCEPTS AVANCÉS

### 6.1 Attention Multi-Têtes

Formule générale :
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
MultiHead(Q, K, V) = Concat(head₁, ..., head_h) W^O
où head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 6.2 Convolution avec Dilatation

```
y[i] = Σ_k w[k] * x[i + k*d]
```
où d = taux de dilatation

### 6.3 Weight Normalization

```
w = g * v / ||v||
```
- **v** : vecteur de poids
- **g** : scalaire appris
- Stabilise l'entraînement

---

## 7. EXERCICES PRATIQUES

### Exercice 1 : Calculer la KL divergence
Données : μ₁=0, σ₁=1, μ₂=2, σ₂=0.5
```
KL = log(0.5/1) + (1 + 4)/(2*0.25) - 0.5
   = -0.693 + 10 - 0.5
   = 8.807
```

### Exercice 2 : Reparameterization
Échantillonner z ~ N(3, 4) :
```
ε ~ N(0,1)  # ex: ε = 0.5
z = 3 + 2*0.5 = 4
```

### Exercice 3 : Log-déterminant du Coupling Layer
Si s(x_a) = [1, 2, 3] :
```
log|det(J)| = 1 + 2 + 3 = 6
```

---

## 8. RÉFÉRENCES MATHÉMATIQUES

### Notations
- **⊙** : produit élément par élément (Hadamard)
- **⊕** : concaténation
- **∇** : gradient
- **∂** : dérivée partielle
- **E[·]** : espérance
- **N(μ,σ²)** : distribution normale

### Constantes
- **π** ≈ 3.14159
- **e** ≈ 2.71828
- **log** : logarithme naturel (base e)

---

## CONCLUSION

VITS combine élégamment :
1. **VAE** pour la modélisation probabiliste
2. **Flows** pour l'expressivité
3. **GAN** pour la qualité audio
4. **MAS** pour l'alignement automatique

Les mathématiques sous-jacentes reposent sur :
- Théorie des probabilités
- Optimisation
- Transformations différentiables
- Apprentissage adversarial
