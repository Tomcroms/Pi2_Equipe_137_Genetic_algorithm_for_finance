class Stock:
    def __init__(self, name, expected_return, std_dev, price):
        self.name = name
        self.expected_return = expected_return  # Rendement attendu
        self.std_dev = std_dev  # Ecart-type du rendement
        self.price = price