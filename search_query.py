# search_query.py
# Paso 2: Implementación de la interfaz remota

import xmlrpc.server
from search_interface import SearchInterface

class SearchQuery(SearchInterface):
    """
    Implementación de la interfaz remota SearchInterface.
    Equivalente a la clase SearchQuery que extiende UnicastRemoteObject en Java.
    """

    def __init__(self):
        """
        Constructor por defecto.
        Equivalente al constructor que puede lanzar RemoteException en Java.
        """
        super().__init__()

    def query(self, search_term):
        """
        Implementación del método de búsqueda.

        Args:
            search_term (str): Término a buscar

        Returns:
            str: "Found" si encuentra "Reflection in Java", "Not Found" en caso contrario

        Raises:
            Exception: Si ocurre un error durante la búsqueda
        """
        try:
            if search_term == "Reflection in Java":
                result = "Found"
            else:
                result = "Not Found"
            return result
        except Exception as e:
            raise Exception(f"Error en búsqueda remota: {str(e)}")

    def register_methods(self, server):
        """
        Registra los métodos que estarán disponibles remotamente.

        Args:
            server: Servidor XML-RPC donde registrar los métodos
        """
        server.register_function(self.query, 'query')