# search_interface.py
# Paso 1: Definición de la interfaz remota en Python

from abc import ABC, abstractmethod

class SearchInterface(ABC):
    """
    Interfaz remota equivalente al Remote interface de Java RMI.
    Define los métodos que pueden ser invocados remotamente.
    """

    @abstractmethod
    def query(self, search_term):
        """
        Método abstracto que debe ser implementado por las clases derivadas.

        Args:
            search_term (str): Término de búsqueda

        Returns:
            str: Resultado de la búsqueda

        Raises:
            Exception: Excepción remota equivalente a RemoteException
        """
        pass