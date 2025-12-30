# search_server.py
# Paso 5: Programa de aplicación del servidor

import xmlrpc.server
import threading
import time
from search_query import SearchQuery

class SearchServer:
    """
    Servidor de aplicación que maneja las solicitudes remotas.
    Equivalente a la clase SearchServer de Java RMI.
    """

    def __init__(self, host='localhost', port=1900):
        """
        Inicializa el servidor.

        Args:
            host (str): Dirección del servidor (por defecto localhost)
            port (int): Puerto del servidor (por defecto 1900)
        """
        self.host = host
        self.port = port
        self.server = None

    def start_server(self):
        """
        Inicia el servidor XML-RPC.
        Equivalente a createRegistry y rebind de Java RMI.
        """
        try:
            # Crear el servidor XML-RPC
            # Equivalente a LocateRegistry.createRegistry(1900)
            self.server = xmlrpc.server.SimpleXMLRPCServer((self.host, self.port),
                                                          allow_none=True,
                                                          logRequests=True)

            # Crear objeto de la implementación de la interfaz
            # Equivalente a Search obj = new SearchQuery()
            search_obj = SearchQuery()

            # Registrar los métodos del objeto remoto
            # Equivalente a Naming.rebind()
            search_obj.register_methods(self.server)

            # Habilitar métodos introspectivos del sistema (opcional)
            self.server.register_introspection_functions()

            print(f"Servidor iniciado en {self.host}:{self.port}")
            print("Esperando solicitudes de clientes...")
            print("Presiona Ctrl+C para detener el servidor")

            # Mantener el servidor ejecutándose
            # Equivalente al comportamiento del servidor RMI
            self.server.serve_forever()

        except KeyboardInterrupt:
            print("\nDeteniendo servidor...")
            if self.server:
                self.server.shutdown()
        except Exception as e:
            print(f"Error en el servidor: {str(e)}")

    def stop_server(self):
        """
        Detiene el servidor.
        """
        if self.server:
            self.server.shutdown()
            print("Servidor detenido")

def main():
    """
    Función principal del servidor.
    """
    try:
        # Crear y iniciar el servidor
        server = SearchServer()
        server.start_server()
    except Exception as e:
        print(f"Error al inicializar el servidor: {str(e)}")

if __name__ == "__main__":
    main()