#include "problemcec2014.h"
#include "problem.h"
#include "algoritmo.h"
#include "random.h"
#include "srandom.h"
#include "domain.h"
#include "localsearch.h"
#include <iostream>
#include <stdlib.h> 
#include <algorithm>  
#include <math.h>  

using namespace realea;

int main(int argc, char *argv[]) {
   //int seed = time(NULL);    // Esto solo se hará una vez

	int dim = 10;
	int n_fun = 1;
	int seed = 10;
	int tope = 0;
	string algoritmo;

	if(argc == 1){
		do{
			cout << "Introducir dimensión (10,30): ";
			cin >> dim;
		}while(dim != 10 && dim != 30 && dim != 50);

		do{
			cout << "Introducir función (1,2, ..., 20): ";
			cin >> n_fun;
		}while(n_fun > 20 || n_fun < 1);

		do{
			cout << "Introducir algoritmo (GSO, GSO_CMAES): ";
			cin >> algoritmo;
		}while(algoritmo != "GSO" && algoritmo != "GSO_CMAES");
	}
	else{
		dim = atoi(argv[1]);
		n_fun = atoi(argv[2]);
		algoritmo = argv[3];

		if(dim != 10 && dim != 30 && dim != 50){
			cout << "ERROR: dim incorrecta (10,30,50)" << endl;
			return 0;
		}
		else if(n_fun > 20 || n_fun < 1){
			cout << "ERROR: n_fun incorrecto (1,2, ..., 20)" << endl;
			return 0;
		}
		else if(algoritmo != "GSO" && algoritmo != "GSO_CMAES"){
			cout << "ERROR: algoritmo incorrecto (GSO, GSO_CMAES)" << endl;
			return 0;
		}
	}  

	SRandom * sr = new SRandom(seed);
	Random random(sr);    
	ProblemCEC2014 cec2014(dim);   
	ProblemPtr problem = cec2014.get(n_fun);   
	DomainRealPtr domain = problem->getDomain();    
	tChromosomeReal sol(dim);
	getInitRandom(&random, domain, sol);    
	Algoritmo * alg = new Algoritmo();    
	alg->setRandom(&random);              
	alg->setProblem(problem.get());       

	tFitness fitness = problem->eval(sol);

	int n = 25;

	vector<double> resultados;
	vector<double> evaluaciones;

	for(int i = 0; i < n; i++){
		tChromosomeReal s_ = sol;
		tFitness f_ = fitness;
		unsigned evals;

		if(algoritmo == "GSO")
			evals = alg->apply(s_, f_, tope);  
		else
			evals = alg->apply2(s_, f_, tope);  

		resultados.push_back(f_ - 100.0*n_fun);
		evaluaciones.push_back(evals);      
	}

	// Calcular medias
	double media = 0;
	double media_eval = 0;

	for(int i = 0; i < n; i++){
		media += resultados[i];
		media_eval += evaluaciones[i];
	}

	media = media/n;
	media_eval = media_eval/n;

	// Calcular desviación
	double desviacion = 0;

	for(int i = 0; i < n; i++){
		double dif = resultados[i]-media;
		desviacion += dif*dif;
	}

	desviacion = sqrt(desviacion/n);

	// Calcular mediana
	sort(resultados.begin(), resultados.end());

	double mediana = resultados[n/2];

	if(n_fun < 10)
		cout << algoritmo << "  Dim = " << dim << ", Funcion =  " << n_fun << "-> Media: " << media << "  Desv: " << desviacion 
				<< "  Mediana: " << mediana << "  Media Evals: " << media_eval << endl;
	else
		cout << algoritmo << "  Dim = " << dim << ", Funcion = " << n_fun << "-> Media: " << media << "  Desv: " << desviacion 
			<< "  Mediana: " << mediana << "  Media Evals: " << media_eval << endl;


  return 0;
}

