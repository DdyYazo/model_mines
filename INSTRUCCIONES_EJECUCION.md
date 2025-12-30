# Instrucciones de Ejecución - Sistema de Métodos Remotos en Python

## Resumen del Proyecto

Este proyecto implementa un sistema de métodos remotos en Python equivalente al ejemplo Java RMI proporcionado. Utiliza XML-RPC para la comunicación entre cliente y servidor.

## Archivos Creados

1. **search_interface.py** - Interfaz remota (equivalente al Remote interface de Java)
2. **search_query.py** - Implementación de la interfaz remota (equivalente a SearchQuery)
3. **search_server.py** - Aplicación del servidor (equivalente a SearchServer)
4. **client_request.py** - Aplicación del cliente (equivalente a ClientRequest)
5. **rmi_requirements.txt** - Dependencias del proyecto

## Requisitos del Sistema

- **Python 3.6 o superior**
- **Librerías estándar de Python** (incluidas por defecto)
  - xmlrpc.server
  - xmlrpc.client
  - abc
  - threading
  - sys
  - time

## Instrucciones de Ejecución

### Paso 1: Verificar Python
```bash
python --version
```

### Paso 2: Ejecutar el Servidor
Abrir una terminal/símbolo del sistema y ejecutar:
```bash
python search_server.py
```

**Salida esperada del servidor:**
```
Servidor iniciado en localhost:1900
Esperando solicitudes de clientes...
Presiona Ctrl+C para detener el servidor
```

### Paso 3: Ejecutar el Cliente
Abrir **OTRA** terminal/símbolo del sistema (mientras el servidor sigue ejecutándose) y ejecutar:
```bash
python client_request.py
```

**Salida esperada del cliente:**
```
Cliente de búsqueda remota
Conectando a: http://localhost:1900
Conectado exitosamente al servidor: http://localhost:1900

=== Realizando consultas de búsqueda ===

Buscando: 'Reflection in Java'
Artículo sobre 'Reflection in Java': Found

Buscando: 'Python Programming'
Artículo sobre 'Python Programming': Not Found

Buscando: 'Machine Learning'
Artículo sobre 'Machine Learning': Not Found

Buscando: 'Remote Method Invocation'
Artículo sobre 'Remote Method Invocation': Not Found

=== Modo interactivo ===
Escribe 'quit' para salir

Ingresa término de búsqueda:
```

### Paso 4: Pruebas Interactivas
En el modo interactivo del cliente, puedes probar diferentes búsquedas:

- Escribe: `Reflection in Java` → Resultado: `Found`
- Escribe: `Cualquier otra cosa` → Resultado: `Not Found`
- Escribe: `quit` → Para salir del cliente

## Equivalencias con Java RMI

| Concepto Java RMI | Implementación Python |
|-------------------|------------------------|
| Remote interface | SearchInterface (ABC) |
| UnicastRemoteObject | XMLRPCServer |
| RemoteException | Exception |
| rmic (stub/skeleton) | XML-RPC automático |
| rmiregistry | XMLRPCServer |
| Naming.rebind() | server.register_function() |
| Naming.lookup() | xmlrpc.client.ServerProxy() |

## Conexión desde Otra Máquina

Para conectar desde otra máquina, modifica la URL del cliente:

```bash
python client_request.py http://IP_DEL_SERVIDOR:1900
```

Donde `IP_DEL_SERVIDOR` es la dirección IP de la máquina donde ejecutas el servidor.

## Detener la Ejecución

- **Servidor**: Presiona `Ctrl+C` en la terminal del servidor
- **Cliente**: Escribe `quit` o presiona `Ctrl+C` en la terminal del cliente

## Solución de Problemas

### Error "Connection refused"
- Verifica que el servidor esté ejecutándose
- Asegúrate de que el puerto 1900 esté disponible

### Error "Address already in use"
- El puerto 1900 ya está en uso
- Detén el servidor anterior o cambia el puerto en ambos archivos

### Error de importación
- Verifica que todos los archivos estén en la misma carpeta
- Verifica la versión de Python (debe ser 3.6+)

## Funcionalidades Implementadas

✅ Interfaz remota abstracta
✅ Implementación del servicio
✅ Servidor XML-RPC
✅ Cliente con modo interactivo
✅ Manejo de excepciones
✅ Consultas automáticas de ejemplo
✅ Soporte para conexiones remotas
✅ Documentación completa

## Diferencias con Java RMI

1. **Sin compilación**: Python no requiere compilar stubs/skeletons
2. **XML-RPC vs RMI**: Usa XML sobre HTTP en lugar del protocolo RMI
3. **Manejo automático**: XML-RPC maneja automáticamente la serialización
4. **Más simple**: Menos configuración que Java RMI

Este sistema proporciona la misma funcionalidad que el ejemplo Java RMI pero adaptado al ecosistema Python.