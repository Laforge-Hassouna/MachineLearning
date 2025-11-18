from abc import ABC, abstractmethod

class Method(ABC):
    @abstractmethod
    def predict(self, X):
        pass
