	Práctica alternativa al examen de teoría - MH
		herrera@decsai.ugr.es

# Sobre este documento #


El proyecto requiere:

- g++. 
- Make. 
- CMake.

Para compilar es necesario hacer:

`$ cmake .`
`$ make`

### Entorno de depuración ###

Por defecto compila para ejecutar más rápidamente, si se desea depurar es necesario hacer:

cmake -DCMAKE_BUILD_TYPE=Debug .

y luego compilar de nuevo con *make*.

## Ejecución ##

Para la ejecución:

`$ ./main`

Ejemplo de ejecución:

$ ./main
Introducir dimensión (10,30): 10
Introducir función (1,2, ..., 20): 5
Introducir algoritmo (GSO, GSO_CMAES): GSO_CMAES
GSO_CMAES  Dim = 10, Funcion =  5-> Media: 20.0682  Desv: 0.0974575  Mediana: 20  Media Evals: 102821

Adicionalmente, es necesario el uso de las siguientes clases:

- Random, para que los métodos de BL puedan generar los números aleatorios. 
- Domain, representa el espacio de búsqueda, usado para comprobar que
  las soluciones se encuentren siempre dentro de la búsqueda.
- Problem, para que la BL pueda evaluar soluciones, y conocer el espacio de búsqueda. 
- ProblemCEC2014, que permite obtener las distintas funciones (o problemas) del benchmark CEC'2014.

