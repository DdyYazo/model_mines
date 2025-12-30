# client_request.py
# Paso 6: Programa de aplicación del cliente

import xmlrpc.client
import sys

class ClientRequest:
    """
    Cliente que realiza solicitudes al servidor remoto.
    Equivalente a la clase ClientRequest de Java RMI.
    """

    def __init__(self, server_url="http://localhost:1900"):
        """
        Inicializa el cliente.

        Args:
            server_url (str): URL del servidor remoto
        """
        self.server_url = server_url
        self.proxy = None

    def connect_to_server(self):
        """
        Establece conexión con el servidor remoto.
        Equivalente a Naming.lookup() de Java RMI.

        Returns:
            bool: True si la conexión es exitosa, False en caso contrario
        """
        try:
            # Crear proxy del objeto remoto
            # Equivalente a Naming.lookup("rmi://localhost:1900/geeksforgeeks")
            self.proxy = xmlrpc.client.ServerProxy(self.server_url, allow_none=True)

            # Probar la conexión con nuestro método personalizado
            # En lugar de system.listMethods(), usamos nuestro método query
            test_result = self.proxy.query("test_connection")
            print(f"Conectado exitosamente al servidor: {self.server_url}")
            return True

        except Exception as e:
            print(f"Error al conectar con el servidor: {str(e)}")
            return False

    def search_query(self, search_term):
        """
        Realiza una consulta de búsqueda al servidor remoto.

        Args:
            search_term (str): Término a buscar

        Returns:
            str: Resultado de la búsqueda
        """
        try:
            if not self.proxy:
                if not self.connect_to_server():
                    return "Error: No se pudo conectar al servidor"

            # Llamada al método remoto
            # Equivalente a access.query(value)
            result = self.proxy.query(search_term)
            return result

        except Exception as e:
            return f"Error en consulta remota: {str(e)}"

    def run_client(self):
        """
        Ejecuta el cliente y realiza consultas de ejemplo.
        """
        try:
            # Conectar al servidor
            if not self.connect_to_server():
                return

            # Consultas de ejemplo
            search_terms = [
                "Reflection in Java",
                "Python Programming",
                "Machine Learning",
                "Remote Method Invocation"
            ]

            print("\n=== Realizando consultas de búsqueda ===")

            for term in search_terms:
                print(f"\nBuscando: '{term}'")
                answer = self.search_query(term)
                print(f"Artículo sobre '{term}': {answer}")

            # Consulta interactiva
            print("\n=== Modo interactivo ===")
            print("Escribe 'quit' para salir")

            while True:
                try:
                    user_input = input("\nIngresa término de búsqueda: ").strip()

                    if user_input.lower() in ['quit', 'exit', 'salir']:
                        print("Cerrando cliente...")
                        break

                    if user_input:
                        result = self.search_query(user_input)
                        print(f"Resultado: {result}")
                    else:
                        print("Por favor ingresa un término válido")

                except KeyboardInterrupt:
                    print("\nCerrando cliente...")
                    break

        except Exception as e:
            print(f"Error en el cliente: {str(e)}")

def main():
    """
    Función principal del cliente.
    """
    # Verificar argumentos de línea de comandos
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    else:
        server_url = "http://localhost:1900"

    print(f"Cliente de búsqueda remota")
    print(f"Conectando a: {server_url}")

    # Crear y ejecutar cliente
    client = ClientRequest(server_url)
    client.run_client()

if __name__ == "__main__":
    main()