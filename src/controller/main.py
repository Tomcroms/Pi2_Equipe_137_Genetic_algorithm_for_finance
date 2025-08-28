import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)


from src.model.genetic_algorithm import GeneticAlgorithm
from src.model.data_loader import DataLoader
import time
import tkinter as tk
from src.view.gui import GAParameterView


def main():
    file_path = 'data/Cac40_Prices_2000_to_Today.xlsx'
    data_loader = DataLoader(file_path)
    
    try:
        stocks, cov_matrix = data_loader.get_data()
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Create a root window for the parameter view
    root = tk.Tk()
    root.withdraw()  # hide the main window

    # Show the GAParameterView as a modal dialog
    param_view = GAParameterView(master=root)
    root.wait_window(param_view)

    if not param_view.parameters_confirmed:
        print("No parameters selected. Exiting.")
        return

    params = param_view.get_parameters()

    # Now instantiate and run the GA with chosen parameters
    try:
        ga = GeneticAlgorithm(
            stocks,
            cov_matrix,
            population_size=params["population_size"],
            is_short_available=params["is_short_available"],
            fitness_function=params["fitness_function"],
            crossover_function=params["crossover_function"],
            mutation_function=params["mutation_function"],
            selection_method=params["selection_method"],
            risk_aversion=params["risk_aversion"],
            budget=params["budget"],
            max_generations=params["max_generations"]
        )

        best_portfolio = ga.evolve(fitness_threshold=15)

        print("Best portfolio:")
        for stock, shares in zip(best_portfolio.stocks, best_portfolio.shares):
            print(f"{stock.name}: {shares} shares")
        print(f"Expected Return (calculate_expected_return_by_percentage): {(best_portfolio.calculate_expected_return()*100)}%")
        print(f"Variance: {best_portfolio.get_volatility_percentage():.2f}")

        time.sleep(1000)

    except Exception as e:
        print(f"Erreur inconnue:\n{e}")
        
if __name__ == "__main__":
    main()