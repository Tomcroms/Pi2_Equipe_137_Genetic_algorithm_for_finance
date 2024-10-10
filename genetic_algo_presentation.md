# Optimisation de Portefeuille avec Algorithme Génétique

Ce projet implémente un algorithme génétique pour optimiser un portefeuille selon le modèle de Markowitz, en utilisant une fonction d'utilité quadratique. Différentes techniques d'optimisation ont été implémentées, telles que la sélection par tournoi avec élitisme, le Simulated Binary Crossover (SBX), et la mutation gaussienne avec normalisation. 

L'algorithme permet d'équilibrer le rendement espéré du portefeuille et le risque associé en fonction d'une aversion au risque donnée.

## Méthodes Utilisées

### Sélection par Tournoi avec Élites

Nous avons choisi la **sélection par tournoi**, où un sous-ensemble d'individus est sélectionné et le meilleur parmi eux est choisi pour la reproduction. Ce type de sélection permet de contrôler directement la pression sélective à travers la taille du tournoi \( k \), ce qui est utile pour maintenir la diversité génétique tout en favorisant les individus les plus performants.

En plus de la sélection par tournoi, nous avons utilisé un mécanisme d'**élitisme** où les meilleurs individus sont directement transférés à la génération suivante, garantissant ainsi que les solutions de qualité ne sont pas perdues d'une génération à l'autre.

### Simulated Binary Crossover (SBX)

Le **Simulated Binary Crossover (SBX)** a été choisi pour la recombinaison des portefeuilles. Contrairement aux méthodes de croisement classiques (comme le crossover à un ou deux points), SBX permet une meilleure exploration des solutions continues. SBX génère des enfants qui sont proches des parents dans l'espace des solutions, tout en introduisant de la variabilité.

\documentclass{article}
\usepackage{amsmath}

\begin{document}

Le \textbf{Simulated Binary Crossover (SBX)} est une méthode de croisement utilisée dans les algorithmes génétiques pour les problèmes d’\textbf{optimisation continue}. Il génère des solutions enfants proches des parents tout en introduisant de la variabilité. SBX est contrôlé par un paramètre clé, \(\eta_{\text{SBX}}\), qui détermine la distance des enfants par rapport aux parents dans l’espace de recherche.

\begin{itemize}
    \item \textbf{Faible \(\eta\) (2-5)} : Favorise l’\textbf{exploration globale} en générant des enfants plus éloignés des parents, ce qui est utile en début d’optimisation ou lorsque l’espace de recherche est vaste.
    
    \item \textbf{Valeur modérée de \(\eta\) (5-15)} : Permet un \textbf{équilibre} entre exploration et exploitation, adapté aux phases intermédiaires de l'algorithme.
    
    \item \textbf{Grand \(\eta\) (15-25)} : Favorise l’\textbf{exploitation locale} en générant des enfants très proches des parents, idéal pour raffiner les solutions lors des phases finales.
\end{itemize}

Le choix de \(\eta\) dépend de la phase de l’algorithme, de la \textbf{taille de la population} et de la \textbf{complexité du problème}. Dans l'optimisation de portefeuille, où les variables sont des proportions continues des actifs, un faible \(\eta\) favorise l’exploration initiale de différentes allocations, tandis qu’un \(\eta\) plus élevé permet de peaufiner les solutions en fin d’optimisation.

\end{document}


### Mutation Gaussienne

Pour maintenir la diversité dans la population, nous avons utilisé une **mutation gaussienne**. Chaque poids du portefeuille est modifié par une petite valeur aléatoire tirée d'une distribution normale. Cette méthode permet de faire varier les poids de manière contrôlée et subtile, ce qui est particulièrement adapté aux problèmes d'optimisation continue comme celui des portefeuilles d'actions.

Après la mutation, les poids sont **normalisés** pour que la somme soit toujours égale à 1, respectant ainsi les contraintes du portefeuille.

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

- **Mutation uniforme** : Chaque poids peut être remplacé par une nouvelle valeur choisie de manière aléatoire dans l'espace de solution.
- **Mutation non uniforme** : La probabilité de mutation diminue au fur et à mesure que l'algorithme progresse, favorisant une exploration large au début, suivie d'une exploitation plus fine des solutions prometteuses.

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
