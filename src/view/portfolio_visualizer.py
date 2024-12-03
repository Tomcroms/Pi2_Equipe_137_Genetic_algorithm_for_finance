import matplotlib.pyplot as plt
import numpy as np

class PortfolioVisualizer:
    def __init__(self):
        plt.ion()  # Activate interactive mode
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Standard Deviation (%)')
        self.ax.set_ylabel('Expected Return (%)')
        self.scatter = self.ax.scatter([], [], s=[], c=[])
        self.portfolios = []  # List to store portfolio data
        # Remove any calls to self.fig.show()
    
    def update(self, portfolios):
        # Increase the age of existing portfolios
        for p in self.portfolios:
            p['age'] += 1

        # Add new portfolios with age 0
        for portfolio in portfolios:
            std_dev = portfolio.get_standard_deviation_percentage()
            expected_return = portfolio.calculate_expected_return_by_percentage()
            self.portfolios.append({
                'std_dev': std_dev,
                'expected_return': expected_return,
                'age': 0
            })

        # Prepare data for plotting
        sizes = []
        colors = []
        xdata = []
        ydata = []
        for p in self.portfolios:
            age = p['age']
            # Size and color calculations (unchanged)
            max_size = 200
            min_size = 20
            decay_rate = 0.5
            size = max_size * np.exp(-decay_rate * age)
            size = max(size, min_size)
            sizes.append(size)
            max_age = 5
            normalized_age = min(age / max_age, 1.0)
            red_intensity = (1 - normalized_age)
            color = (red_intensity, 0, 0)
            colors.append(color)
            xdata.append(p['std_dev'])
            ydata.append(p['expected_return'])

        # print("xdata:", xdata)
        # print("ydata:", ydata)

        # Update the scatter plot
        self.scatter.set_offsets(np.column_stack((xdata, ydata)))
        self.scatter.set_sizes(sizes)
        self.scatter.set_color(colors)

        # Manually set axes limits based on data
        x_min, x_max = min(xdata) * 0.95, max(xdata) * 1.05
        y_min, y_max = min(ydata) * 0.95, max(ydata) * 1.05
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

        # Redraw the plot
        plt.draw()
        plt.pause(0.001)