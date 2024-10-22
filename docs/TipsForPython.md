# Tips for Python 


## Afficher erreurs de syntaxes

Utiliser pycodestyle nom_du_fichier.py pour afficher les erreurs de syntaxes selon la norme PEP 8 (norme officielle de python).


## Classes 


### Une classe -> un fichier 

### Instanciation immédiate d'un objet 

class Stock:
    def __init__(self, name, expected_return, std_dev):
        self.name = name
        self.expected_return = expected_return  # Rendement attendu
        self.std_dev = std_dev  # Écart-type du rendement

stockConfig = {"name": "ActionB", "expected_return": 0.6, "std_dev": 22}
s = Stock(**stockConfig)    #instanciation immédiate pour actionB

### Principal intérêt des classes -> surcharges d'opérateurs