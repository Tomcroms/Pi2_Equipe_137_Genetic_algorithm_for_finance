# fitness_visualization.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
from src.model.portfolio import Portfolio

class FitnessVisualizer:
    def __init__(self):
        # Initialize the figure and 3D axes
        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Lists to store data points
        self.expected_returns = []
        self.volatilities = []
        self.fitnesses = []
        self.generations = []
        
        # Scatter plot
        self.scatter = self.ax.scatter([], [], [], c=[], cmap='viridis', marker='o')
        
        # Labels
        self.ax.set_xlabel('Expected Return (%)')
        self.ax.set_ylabel('Volatility (%)')
        self.ax.set_zlabel('Fitness Function')
        self.ax.set_title('3D Fitness Progression Over Generations')
        
        # Initialize color mapping for fitness
        self.cmap = plt.get_cmap('viridis')
        self.norm = plt.Normalize()
        
        # For dynamic updating
        plt.ion()
        plt.show()

    def update(self, portfolio: Portfolio, generation):
        """
        Update the visualization with the best portfolio's metrics.
        
        Args:
            portfolio (Portfolio): The best portfolio object of the current generation.
            generation (int): The current generation number.
        """
        # Extract metrics
        exp_return = portfolio.get_expected_return_percentage()
        volatility = portfolio.get_volatility_percentage()
        fitness = portfolio.fitness
        
        # Append to lists
        self.expected_returns.append(exp_return)
        self.volatilities.append(volatility)
        self.fitnesses.append(fitness)
        self.generations.append(generation)
        
        # Update scatter data
        self.ax.scatter(exp_return, volatility, fitness, c=[fitness], cmap='viridis', norm=self.norm, marker='o')
        
        # Update color normalization
        self.norm.autoscale(self.fitnesses)
        self.scatter.set_array(np.array(self.fitnesses))
        
        # Adjust the view limits
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show(self):
        """Display the plot in non-interactive mode."""
        plt.ioff()
        plt.show()

    def save_animation(self, filename='fitness_progress.mp4', fps=2):
        """
        Save the visualization as an animation.
        
        Args:
            filename (str): The filename for the saved animation.
            fps (int): Frames per second for the animation.
        """
        # Create a new figure for the animation
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Expected Return (%)')
        ax.set_ylabel('Volatility (%)')
        ax.set_zlabel('Fitness Function')
        ax.set_title('3D Fitness Progression Over Generations')
        
        scat = ax.scatter([], [], [], c=[], cmap='viridis')
        norm = plt.Normalize(min(self.fitnesses), max(self.fitnesses))
        
        def init():
            scat._offsets3d = ([], [], [])
            scat.set_array([])
            return scat,
        
        def animate(i):
            scat._offsets3d = (self.expected_returns[:i+1], self.volatilities[:i+1], self.fitnesses[:i+1])
            scat.set_array(np.array(self.fitnesses[:i+1]))
            return scat,
        
        ani = animation.FuncAnimation(fig, animate, init_func=init,
                                      frames=len(self.generations), interval=500, blit=False)
        
        ani.save(filename, writer='ffmpeg', fps=fps)
        print(f"Animation saved as {filename}")
