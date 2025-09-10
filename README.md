# Portfolio Optimization with Genetic Algorithm

This project implements a genetic algorithm to optimize a portfolio according to the Markowitz model, using a quadratic utility function. Various optimization techniques have been implemented, such as tournament selection with elitism, Simulated Binary Crossover (SBX), and Gaussian mutation with normalization.

The algorithm balances the expected return of the portfolio and the associated risk, depending on a given risk aversion.

### See /docs/docs.md for more details on genetic algorithms

# Genetic Algorithm Implementation TEAM 137

Here are the different methods we chose for the various stages of our genetic algorithm.

## 1. Population Initialization

Let $n$ be the number of assets available in the portfolio  
Let $p$ be the price of the asset  
Let $B$ be the defined budget  

For each portfolio in the population (each portfolio being a combination of assets and quantities), we generate a vector of $n$ random uniform values $s_i ∈ [0,1]$.

Then, for each of these random values:

$s_i = B/p_i$ → These values are multiplied by the maximum number of shares that can be purchased for each asset $i$, within the budget limit.  

Values are then rounded down (floor function) to ensure that the number of shares purchased is an integer.  

At initialization, the portfolio budget may exceed the total budget. However, this is not an issue, since the objective function penalizes budget overruns. Over time, the budget converges to the desired level.  
This ensures the initial population of portfolios remains as random as possible.

## 2. Fitness Function

Once the portfolio population is generated, we can compute their fitness functions.  
The portfolio fitness function is calculated as follows:

\[
S = \frac{E[R_p] - R_f}{\sigma_p}
\]

Where:  
- $E[R_p]$ = Expected return of the portfolio  
- $R_f$ = Risk-free rate  
- $\sigma_p$ = Standard deviation of portfolio returns  

## 3. Evolutionary Function

The evolutionary function is straightforward:

We loop until the fitness of the best portfolio exceeds the target objective.

At each iteration:

- Select portfolios for the next generation  
- Perform crossovers on the tournament-selected portfolios  
- Apply mutation to the offspring portfolios  

## 4. Elitism and Tournament Selection

Portfolio selection is done in two steps:

1. **Elitism**: Keep the top 10% of portfolios in the generation unchanged.  
2. **Tournament Selection**: For the remaining 90%, select parents via tournament. Randomly choose 3 portfolios from the current generation, and select the 2 best among them as parents.  

## 5. Crossover

After selecting parent pairs, each pair is crossed as follows:

ETA (𝜂) is the crossover distribution index (a high 𝜂 reduces child dispersion).  

For each asset:  
Generate a random number $u ∈ [0,1]$.  
The new share is computed as a weighted average of the shares from both parents.

Formula:  
\[
\text{child\_share} = 0.5 \cdot \big((1+β)\cdot \text{parent1\_share} + (1-β)\cdot \text{parent2\_share}\big)
\]

- $(1+β)\cdot parent1\_share$ → weights the first parent’s share  
- $(1-β)\cdot parent2\_share$ → weights the second parent’s share  

Thus:  
- If $β \to 1$ → closer to parent 1  
- If $β \to 0$ → closer to parent 2  
- If $β \to 0.5$ → balanced contribution  

The parameter $β$ is calculated as follows:  

If $u \leq 0.5$ :  
![Equation](img/Beta_u_inf.png)

If $u > 0.5$ :  
![Equation](img/Beta_u_sup.png)

Example numerical application:  
![Example](img/exemple_beta_u.png)

In this example, the values of β for $u=0.25$ and $u=0.75$ are approximate inverses of each other, illustrating symmetry around the mean.

## 6. Mutation

Mutation is then applied to offspring as follows:  

- Mutation probability set to 10%  
- Sigma set to 0.1 (standard deviation of Gaussian noise)  

For each share in the created portfolio:  
- Apply Gaussian perturbation if selected  
  ![Mutation Formula](img/mutation_formula.png)  
- Ensure no negative shares  
- Keep relative share proportions within the budget
