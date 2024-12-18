import tkinter as tk
from tkinter import ttk

class GAParameterView(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Genetic Algorithm Parameters")
        self.geometry("500x600")

        # Variables
        self.population_size_var = tk.IntVar(value=100)
        self.is_short_available_var = tk.BooleanVar(value=False)
        self.fitness_function_var = tk.StringVar(value="sharpe ratio")
        self.crossover_function_var = tk.StringVar(value="simulated binary crossover")
        self.mutation_function_var = tk.StringVar(value="gaussian mutation")
        self.selection_method_var = tk.StringVar(value="tournament selection")
        self.risk_aversion_var = tk.IntVar(value=6)
        self.budget_var = tk.StringVar(value="1000000000")
        self.max_generations_var = tk.StringVar(value="1000")

        self.create_widgets()

        # Variable to signal that the user pressed Start
        self.parameters_confirmed = False

    def create_widgets(self):
        # Population Size
        ttk.Label(self, text="Population Size:").pack(anchor=tk.W, pady=(10,0), padx=10)
        pop_scale = tk.Scale(self, from_=50, to=200, orient=tk.HORIZONTAL, 
                            variable=self.population_size_var, resolution=1, showvalue=True)
        pop_scale.pack(fill=tk.X, padx=10)

        # Is Short Available
        ttk.Label(self, text="Short Available:").pack(anchor=tk.W, pady=(10,0), padx=10)
        short_frame = tk.Frame(self)
        short_frame.pack(anchor=tk.W, padx=10)
        tk.Checkbutton(short_frame, text="Yes/No", variable=self.is_short_available_var).pack(anchor=tk.W)

        # Fitness Function
        ttk.Label(self, text="Fitness Function:").pack(anchor=tk.W, pady=(10,0), padx=10)
        fitness_cb = ttk.Combobox(
            self, textvariable=self.fitness_function_var,
            values=["sharpe ratio", "quadratic utility"]
        )
        fitness_cb.pack(fill=tk.X, padx=10)

        # Crossover Function
        ttk.Label(self, text="Crossover Function:").pack(anchor=tk.W, pady=(10,0), padx=10)
        crossover_cb = ttk.Combobox(
            self, textvariable=self.crossover_function_var,
            values=["simulated binary crossover", "blend crossover", "arithmetic crossover"]
        )
        crossover_cb.pack(fill=tk.X, padx=10)

        # Mutation Function
        ttk.Label(self, text="Mutation Function:").pack(anchor=tk.W, pady=(10,0), padx=10)
        mutation_cb = ttk.Combobox(
            self, textvariable=self.mutation_function_var,
            values=["gaussian mutation", "polynomial mutation", "uniform mutation"]
        )
        mutation_cb.pack(fill=tk.X, padx=10)

        # Selection Method
        ttk.Label(self, text="Selection Method:").pack(anchor=tk.W, pady=(10,0), padx=10)
        selection_cb = ttk.Combobox(
            self, textvariable=self.selection_method_var,
            values=["tournament selection"], state="readonly"
        )
        selection_cb.pack(fill=tk.X, padx=10)

        # Risk Aversion
        ttk.Label(self, text="Risk Aversion:").pack(anchor=tk.W, pady=(10,0), padx=10)
        risk_scale = tk.Scale(self, from_=1, to=10, orient=tk.HORIZONTAL, 
                    variable=self.risk_aversion_var, resolution=1, showvalue=True)
        risk_scale.pack(fill=tk.X, padx=10)

        # Budget
        ttk.Label(self, text="Budget:").pack(anchor=tk.W, pady=(10,0), padx=10)
        budget_entry = ttk.Entry(self, textvariable=self.budget_var)
        budget_entry.pack(fill=tk.X, padx=10)

        # Max Generations
        ttk.Label(self, text="Max Generations:").pack(anchor=tk.W, pady=(10,0), padx=10)
        max_gen_entry = ttk.Entry(self, textvariable=self.max_generations_var)
        max_gen_entry.pack(fill=tk.X, padx=10, pady=(0,10))

        # Start Button
        start_button = ttk.Button(self, text="Start", command=self.on_start)
        start_button.pack(pady=(20,10))

    def on_start(self):
        # User pressed Start
        self.parameters_confirmed = True
        self.destroy()

    def get_parameters(self):
        return {
            "population_size": self.population_size_var.get(),
            "is_short_available": self.is_short_available_var.get(),
            "fitness_function": self.fitness_function_var.get(),
            "crossover_function": self.crossover_function_var.get(),
            "mutation_function": self.mutation_function_var.get(),
            "selection_method": self.selection_method_var.get(),
            "risk_aversion": self.risk_aversion_var.get(),
            "budget": int(self.budget_var.get()),
            "max_generations": int(self.max_generations_var.get())
        }



