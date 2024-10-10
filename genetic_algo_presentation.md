# Optimisation de Portefeuille avec Algorithme Génétique

Ce projet implémente un algorithme génétique pour optimiser un portefeuille selon le modèle de Markowitz, en utilisant une fonction d'utilité quadratique. Différentes techniques d'optimisation ont été implémentées, telles que la sélection par tournoi avec élitisme, le Simulated Binary Crossover (SBX), et la mutation gaussienne avec normalisation. 

L'algorithme permet d'équilibrer le rendement espéré du portefeuille et le risque associé en fonction d'une aversion au risque donnée.

## Méthodes Utilisées

### Sélection par Tournoi avec Élites

Nous avons choisi la **sélection par tournoi**, où un sous-ensemble d'individus est sélectionné et le meilleur parmi eux est choisi pour la reproduction. Ce type de sélection permet de contrôler directement la pression sélective à travers la taille du tournoi \( k \), ce qui est utile pour maintenir la diversité génétique tout en favorisant les individus les plus performants.

En plus de la sélection par tournoi, nous avons utilisé un mécanisme d'**élitisme** où les meilleurs individus sont directement transférés à la génération suivante, garantissant ainsi que les solutions de qualité ne sont pas perdues d'une génération à l'autre.

Description détaillé de la sélection par tournoi avec Elites:

Conservation des meilleures solutions (élitisme) : En optimisation de portefeuille, certaines allocations peuvent offrir des compromis très avantageux entre risque et rendement. Grâce à l'élitisme, ces portefeuilles optimaux ne risquent pas d'être perdus par des mécanismes aléatoires de reproduction ou de mutation. Cela garantit que les meilleurs portefeuilles identifiés jusque-là sont toujours disponibles pour des améliorations supplémentaires dans les générations futures.

Pression sélective contrôlée (sélection par tournoi) : La sélection par tournoi permet de contrôler la pression sélective en ajustant la taille du tournoi. Avec un petit tournoi, les individus moins performants ont encore une chance de se reproduire, ce qui maintient une diversité génétique importante. Dans l’optimisation de portefeuille, cela favorise l’exploration de nouvelles combinaisons de poids d’actifs, ce qui peut mener à des solutions innovantes. Avec un tournoi plus grand, la pression sélective augmente, favorisant l’exploitation des meilleures solutions déjà identifiées.

Équilibre entre exploration et exploitation : L'optimisation de portefeuille nécessite un bon équilibre entre exploration (découverte de nouvelles solutions) et exploitation (amélioration des solutions existantes). La sélection par tournoi avec élites permet ce compromis : d’un côté, l’élitisme garantit que les meilleures solutions sont conservées, favorisant l'exploitation, tandis que la sélection par tournoi permet de maintenir une diversité de solutions, stimulant l'exploration.

Résistance au bruit des marchés financiers : Dans l'optimisation de portefeuille, les petites variations des données de marché (par exemple, les rendements futurs des actifs) peuvent rendre le problème volatile. La sélection par tournoi avec élites introduit une robustesse en conservant les meilleures solutions sur plusieurs générations, tout en introduisant une diversité contrôlée qui peut réagir aux changements du marché.

Comparaison avec d’autres méthodes de sélection :
Roulette Wheel Selection (sélection proportionnelle) : Cette méthode sélectionne les individus en fonction de la proportion de leur fitness par rapport à la population. Cependant, elle peut favoriser les solutions de manière disproportionnée, menant à une perte de diversité trop rapide. Dans l'optimisation de portefeuille, cela pourrait entraîner une convergence prématurée vers des solutions sous-optimales.

Random Selection (sélection aléatoire) : Elle maintient une bonne diversité, mais le processus étant complètement aléatoire, les meilleures solutions peuvent être facilement perdues. Cela n'est pas souhaitable dans un contexte d'optimisation de portefeuille, où il est crucial de conserver les allocations les plus prometteuses.

Rank Selection (sélection par rang) : Cette méthode sélectionne les individus en fonction de leur rang dans la population plutôt que directement en fonction de leur fitness. Elle est moins sensible aux écarts de fitness entre individus, mais manque de pression sélective directe comparée à la sélection par tournoi, où les meilleurs individus sont activement mis en compétition.

Conclusion :
La sélection par tournoi avec élites est un compromis intelligent pour l'optimisation de portefeuille, car elle combine la robustesse de la sélection des meilleures solutions (grâce à l'élitisme) avec la flexibilité nécessaire pour maintenir une diversité génétique contrôlée (grâce au tournoi). Ce mécanisme permet une convergence progressive et stable vers des portefeuilles optimaux, tout en explorant efficacement l'espace des solutions.

### Simulated Binary Crossover (SBX)

Le **Simulated Binary Crossover (SBX)** a été choisi pour la recombinaison des portefeuilles. Contrairement aux méthodes de croisement classiques (comme le crossover à un ou deux points), SBX permet une meilleure exploration des solutions continues. SBX génère des enfants qui sont proches des parents dans l'espace des solutions, tout en introduisant de la variabilité.

Le **Simulated Binary Crossover (SBX)** est une méthode de croisement utilisée dans les algorithmes génétiques pour les problèmes d’**optimisation continue**. Il génère des solutions enfants proches des parents tout en introduisant de la variabilité. SBX est contrôlé par un paramètre clé, \(\eta_{\text{SBX}}\), qui détermine la distance des enfants par rapport aux parents dans l’espace de recherche, appelé le paramètre de distribution.

- **Faible \(\eta\) (2-5)** : Favorise l’**exploration globale** en générant des enfants plus éloignés des parents, ce qui est utile en début d’optimisation ou lorsque l’espace de recherche est vaste.

- **Valeur modérée de \(\eta\) (5-15)** : Permet un **équilibre** entre exploration et exploitation, adapté aux phases intermédiaires de l'algorithme.

- **Grand \(\eta\) (15-25)** : Favorise l’**exploitation locale** en générant des enfants très proches des parents, idéal pour raffiner les solutions lors des phases finales.

Le choix de \(\eta\) dépend de la phase de l’algorithme, de la **taille de la population** et de la **complexité du problème**. Dans l'optimisation de portefeuille, où les variables sont des proportions continues des actifs, un faible \(\eta\) favorise l’exploration initiale de différentes allocations, tandis qu’un \(\eta\) plus élevé permet de peaufiner les solutions en fin d’optimisation.

## Étapes du SBX

### 1. Sélection des parents

On sélectionne deux parents \( P_1 \) et \( P_2 \), représentés par des vecteurs de `n` variables continues :

- Parent 1 : `P_1 = (x_1^1, x_2^1, ..., x_n^1)`
- Parent 2 : `P_2 = (x_1^2, x_2^2, ..., x_n^2)`

### 2. Calcul du coefficient de distribution `beta_q`

Le **coefficient de distribution** `beta_q` est calculé pour chaque variable \(x_i\) (avec \(i = 1, 2, ..., n\)). Il permet de contrôler la dispersion des enfants autour des parents. La formule est donnée par :

\[
\beta_q(u) = 
\begin{cases} 
(2u)^{\frac{1}{\eta_c+1}}, & \text{si } u \leq 0.5, \\
\left(\frac{1}{2(1-u)}\right)^{\frac{1}{\eta_c+1}}, & \text{sinon}.
\end{cases}
\]


EN markdown 

βq(u) = - Si u <= 0.5: (2u)^(1 / (ηₐ + 1)) - Sinon: [1 / (2(1 - u))]^(1 / (ηₐ + 1))


### Mutation Gaussienne

Pour maintenir la diversité dans la population, nous avons utilisé une **mutation gaussienne**. Chaque poids du portefeuille est modifié par une petite valeur aléatoire tirée d'une distribution normale. Cette méthode permet de faire varier les poids de manière contrôlée et subtile, ce qui est particulièrement adapté aux problèmes d'optimisation continue comme celui des portefeuilles d'actions.

Après la mutation, les poids sont **normalisés** pour que la somme soit toujours égale à 1, respectant ainsi les contraintes du portefeuille.

Pourquoi avoir choisi la mutation gaussienne (et pas un autre type de mutation) ?

1. Adaptation aux problèmes d'optimisation continue :
La mutation gaussienne consiste à ajouter une petite perturbation aléatoire à chaque poids du portefeuille, tirée d'une distribution normale centrée sur 0. Cette méthode permet d'explorer des variations subtiles autour des solutions actuelles, ce qui est particulièrement utile dans un problème d'optimisation continue comme celui d'un portefeuille d'actions, où les variables (les poids des actifs) sont des nombres réels.

Modifications contrôlées : La distribution normale produit souvent des petites perturbations (grâce à la concentration de la distribution autour de la moyenne), ce qui permet de raffiner progressivement les solutions sans perturber fortement les bonnes configurations de poids. Cela est important dans l'optimisation de portefeuille, où de petits changements dans les proportions des actifs peuvent avoir un effet significatif sur la performance globale (rendement et risque).
2. Maintien de la diversité :
La mutation gaussienne introduit de la diversité dans la population en créant des variations dans les poids du portefeuille. Cette diversité est cruciale pour éviter la convergence prématurée de l'algorithme vers un optimum local, ce qui est souvent un problème dans l'optimisation de portefeuille, où il existe de nombreuses configurations locales attrayantes en termes de risque et rendement.

3. Normalisation après mutation :
Dans l'optimisation de portefeuille, la somme des poids doit être égale à 1 (une contrainte de normalisation). Après la mutation, les poids sont donc normalisés, ce qui garantit que l’allocation respecte toujours les contraintes du portefeuille. Cela fait de la mutation gaussienne un choix approprié, car elle permet d'introduire des variations sans violer les contraintes fondamentales du problème.

4. Sensibilité au bruit des marchés :
Les marchés financiers sont souvent soumis à des fluctuations aléatoires, et une mutation gaussienne reflète cette réalité, introduisant des ajustements progressifs dans les poids du portefeuille. Elle permet à l'algorithme de s'adapter à de légers changements, comme ceux observés dans les données financières, tout en explorant des solutions proches de l'optimum.

Pourquoi cela pourrait ne pas être optimal :
Perturbations faibles : Si la distribution normale est trop concentrée autour de 0 (variance faible), les perturbations apportées par la mutation peuvent être trop petites pour échapper aux optima locaux, limitant l'exploration globale.
Adaptation lente : Si les marchés ou les objectifs changent rapidement, des perturbations trop faibles peuvent entraîner une adaptation trop lente du portefeuille, et l'algorithme pourrait avoir du mal à explorer de nouvelles régions de l’espace de solutions de manière significative.

## Autres Méthodes Alternatives

Voici quelques alternatives que nous aurions pu utiliser à différents stades de l'algorithme :

### Sélection

- **Roulette Wheel Selection** : Les portefeuilles sont sélectionnés en fonction de leur fitness proportionnelle. Cela peut toutefois favoriser excessivement les individus très performants.
- **Rank-Based Selection** : Ici, les portefeuilles sont triés par rang, évitant que les solutions dominantes ne prennent trop d'importance dans le processus.

### Combinaison (Crossover)

- **Crossover à un point** : Les poids sont divisés à un point aléatoire, et chaque segment provient d'un des parents.
- **Crossover à deux points** : Les poids sont divisés en trois segments, chacun provenant des parents, ce qui introduit plus de variété dans les combinaisons.

### Fonction de Fitness (Objectif)

- **Ratio de Sharpe** : 
    Le ratio de Sharpe peut être utilisé pour évaluer la performance d'un portefeuille en fonction de son rendement ajusté pour le risque.
    \[
    \text{Sharpe} = \frac{\mathbb{E}[R] - R_f}{\sigma_R}
    \]
    où \( R_f \) est le taux sans risque et \( \sigma_R \) est la volatilité du portefeuille.

- **Ratio de Sortino** : Semblable au ratio de Sharpe, mais il utilise la volatilité des pertes (downsides) au lieu de la volatilité totale.

### Mutation

**Mutation uniforme** :

Principe : La mutation uniforme consiste à remplacer un poids par une valeur aléatoire choisie uniformément dans un intervalle prédéfini (par exemple, entre 0 et 1).
Avantages : Elle peut générer des solutions plus éloignées, offrant une meilleure exploration globale.
Inconvénients : Les perturbations peuvent être trop grandes et drastiques, entraînant des modifications non contrôlées qui pourraient perturber une solution déjà bonne. Cela pourrait rendre difficile la stabilisation des solutions autour d’un optimum local.

**Mutation par "flip bit"** :

Principe : Similaire à une mutation binaire, cette méthode inverse les valeurs ou modifie un poids en le remplaçant par une valeur fixe (ou un seuil prédéfini).
Avantages : Elle est simple et peut être efficace pour explorer rapidement de nouvelles solutions.
Inconvénients : Ce type de mutation est plus adapté aux problèmes discrets et n’est généralement pas efficace pour des problèmes continus comme l'optimisation de portefeuille.

**Mutation non uniforme** :

Principe : Introduit des perturbations qui dépendent du nombre de générations écoulées. Plus l’algorithme avance, plus les mutations deviennent petites, favorisant une convergence fine à la fin.
Avantages : Permet un meilleur contrôle de l’exploration au début et de l’exploitation à la fin, s’adaptant à l’évolution de l’algorithme.
Inconvénients : Peut être plus complexe à paramétrer et pourrait encore conduire à une exploration limitée si mal utilisé.

**Mutation Lévy flight** :

Principe : Cette méthode utilise une distribution de Lévy (une distribution à longue traîne) pour générer des mutations. Cela permet à la mutation de générer des petites perturbations la plupart du temps, mais occasionnellement de faire des perturbations très larges.
Avantages : Elle permet à l’algorithme de sortir de manière plus agressive des optima locaux en introduisant des perturbations importantes, tout en maintenant des ajustements plus subtils la plupart du temps.
Inconvénients : Peut introduire trop d'instabilité si les perturbations sont trop fréquentes ou trop grandes.


## Fonction d'Utilité Quadratique

La fonction d'utilité quadratique est utilisée pour évaluer la performance des portefeuilles. Cette fonction intègre à la fois le rendement espéré et le risque sous forme de variance, tout en tenant compte de l'aversion au risque de l'investisseur.

$$
U = \mathbb{E}[R] - \frac{\lambda}{2} \times \text{Var}(R)
$$

où :
- \( \mathbb{E}[R] \) est le rendement espéré du portefeuille.
- \( \text{Var}(R) \) est la variance (ou risque) du portefeuille.
- \( \lambda \) est le coefficient d'aversion au risque. Ici, nous avons utilisé \( \lambda = 4 \) pour modérer l'équilibre entre le rendement et le risque.

## Extraits de Code

### Sélection par Tournoi

```python
def tournament_selection(self, k=3):
    selected = []
    for _ in range(2):  # Sélectionner deux parents
        individuals = np.random.choice(self.population, k)
        best = max(individuals, key=lambda ind: ind.fitness)
        selected.append(best)
    return selected
