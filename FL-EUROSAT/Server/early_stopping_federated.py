class EarlyStoppingFederated:
    def __init__(self, patience=5, min_delta=0.001):
        """
        Inicializa o Early Stopping para Aprendizado Federado.
        
        Args:
            patience (int): Número de rounds para aguardar melhoria
            min_delta (float): Mínima mudança na métrica para considerar como melhoria
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_accuracy = None
        self.should_stop = False
        
    def __call__(self, current_accuracy):
        """
        Verifica se o treinamento deve parar baseado na acurácia atual.
        
        Args:
            current_accuracy (float): Acurácia agregada do round atual
            
        Returns:
            bool: True se deve parar, False caso contrário
        """
        if self.best_accuracy is None:
            self.best_accuracy = current_accuracy
            return False
            
        if current_accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = current_accuracy
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            return True
            
        return False