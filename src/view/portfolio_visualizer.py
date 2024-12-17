import matplotlib.pyplot as plt
import numpy as np

class PortfolioVisualizer:
    def __init__(self):
        plt.ion()  # Activate interactive mode
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Volatility (%)')
        self.ax.set_ylabel('Expected Return (%)')
        self.scatter = self.ax.scatter([], [], s=[], c=[])
        self.portfolios = []  # List to store portfolio data
    
    def update(self, portfolios, best_portfolio):
        # Increase the age of existing portfolios
        for p in self.portfolios:
            p['age'] += 1

        # On vérifie s'il s'agit du meilleur portefeuille
        for portfolio in portfolios:
            volatility = portfolio.get_volatility_percentage()
            expected_return = portfolio.calculate_expected_return()*100
            is_best = (portfolio == best_portfolio)
            self.portfolios.append({
                'volatility': volatility,
                'expected_return': expected_return,
                'age': 0,
                'is_best': is_best
            })

        # Préparation des données pour le tracé
        sizes = []
        colors = []
        xdata = []
        ydata = []
        
        # Paramètres de base
        max_size = 200
        min_size = 20
        decay_rate = 0.5
        max_age = 5

        for p in self.portfolios:
            age = p['age']
            normalized_age = min(age / max_age, 1.0)

            #Size according to age of portfolio
            size = max_size * np.exp(-decay_rate * age)
            size = max(size, min_size)

            if p['is_best']:
                # Yellow color for best portfolio
                color = (1, 1, 0)
                # And bigger size
                size *= 1.5  
            else:
                # Red color which becomes black when portfolio is older
                red_intensity = (1 - normalized_age)
                color = (red_intensity, 0, 0)

            sizes.append(size)
            colors.append(color)
            xdata.append(p['volatility'])
            ydata.append(p['expected_return'])

        self.scatter.set_offsets(np.column_stack((xdata, ydata)))
        self.scatter.set_sizes(sizes)
        self.scatter.set_color(colors)

        # Ajusting axis
        x_min, x_max = min(xdata) * 0.95, max(xdata) * 1.05
        y_min, y_max = min(ydata) * 0.95, max(ydata) * 1.05
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

        # Re draw
        plt.draw()
        plt.pause(0.001)
