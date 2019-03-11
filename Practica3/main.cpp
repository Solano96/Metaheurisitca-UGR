#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <list>
#include <set>
#include <cstdlib>
#include <limits>

using namespace std;

/************************ Random Number ***************************/

unsigned long Seed = 0L;

#define MASK 2147483647
#define PRIME 65539
#define SCALE 0.4656612875e-9

void Set_random (unsigned long x){
    Seed = (unsigned long) x;
}

unsigned long Get_random (void){
    return Seed;
}

float Rand(void){
    return (( Seed = ( (Seed * PRIME) & MASK) ) * SCALE );
}

int Randint(int low, int high){
    return (int) (low + (high-(low)+1) * Rand());
}

float Randfloat(float low, float high){
    return (low + (high-(low))*Rand());
}

double Normal(double mu, double sigma){
	static const double epsilon = std::numeric_limits<double>::min();
	static const double two_pi = 2.0*3.14159265358979323846;

	thread_local double z1;
	thread_local bool generate;
	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;

	double u1, u2;
	do{
	   u1 = Rand();
	   u2 = Rand();
	}while ( u1 <= epsilon );

	double z0;
	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}

void swap(int &a, int &b){
	int aux = a;
	a = b;
	b = aux;
}

void RandShuffle(vector<int> &v, int first, int last){
	for(int i = first; i < last; i++){
		swap(v[i], v[Randint(first, last-1)]);
	}
}

/************************************************************************/

vector< vector <double> > e;
vector< vector <vector <double> > > partitions(5);
double alpha;

// Read data from arff file
void input(string file_name){
	ifstream fe(file_name); 
	string data;

	set<vector<double> > aux;

	do{
		getline(fe, data, '\n');
		stringstream iss;
		iss << data;
		getline(iss, data, '\r');
	}while(data != "@data");

	while(getline(fe, data, '\n')){
		vector<double> v;
		string cad = "";

		for(int i = 0; i < data.size(); i++){
			if(data[i] != ',')
				cad += data[i];
			else{
				v.push_back(atof(cad.c_str()));
				cad = "";
			}
		}

		v.push_back(atof(cad.c_str()));
		aux.insert(v);
	}

	for(set<vector<double> >::iterator it = aux.begin(); it != aux.end(); it++){
		e.push_back(*it);
	}
}

// Normalize the data (by column)
void Normalize(vector< vector <double> > &T){
	double min, max;
	int n = T.size();
	int num_atrib = T[0].size();

	for(int i = 0; i < num_atrib-1; i++){
		max = min = T[0][i];

		for(int j = 1; j < T.size(); j++){
			if(max < T[j][i]) max = T[j][i];
			if(min > T[j][i]) min = T[j][i];
		}

		for(int j = 0; j < T.size(); j++){
			T[j][i] = (T[j][i] - min)/(max-min);
		}
	}
}

// Calculate the euclidean distance, modified by a weight vector, between two vectors 
double dist(const vector<double> &e1, const vector<double> &e2, const vector<double> &w = vector<double>(e[0].size(),1.0)){
	double sum = 0; 
	int n = e1.size();

	for(int i = 0; i < n-1; i++)
		if(w[i] >= 0.2)
			sum += w[i]*(e1[i]-e2[i])*(e1[i]-e2[i]);

	return sqrt(sum);
}

void createPartitions(){
	int n = e.size();
	int num_atrib = e[0].size();
	int j = 0;

	// Distribute the object rotating the partitions
	// First we distribute the object with label '1'
	for(int i = 0; i < n; i++)
		if(e[i][num_atrib-1] == 1)
			partitions[(j++)%5].push_back(e[i]);

	// Then we distrbute the object with label '2'
	for(int i = 0; i < n; i++)
		if(e[i][num_atrib-1] == 2)
			partitions[(j++)%5].push_back(e[i]);
}

// Generate a vector with ones used for 1-NN algorithm
vector<double> ones(const vector< vector <double> > &T){
	int num_atrib = T[0].size();
	return vector<double>(num_atrib-1, 1);
}

int KNN(const vector< vector <double> > &T, const vector<double> &new_e, const vector<double> &w, int out = -1){
	int num_atrib = T[0].size();
	int c_min = T[0][num_atrib-1];
	double d_min = 99999999;
	int n = T.size();

	for(int i = 0; i < n; i++){
		if(out != i){
			double d = dist(T[i], new_e, w);
			if(d < d_min){
				c_min = T[i][num_atrib-1];
				d_min = d;
			}
		}
	}

	return c_min;
}

/*************************************************************************************/
/*************************************************************************************/

double tasaClass(const vector< vector <double> > &Data, const vector< vector <double> > &T, const vector<double> &w, bool leave_one_out = true){
	int out, n = T.size();
	int num_atrib = T[0].size();
	int clasify_ok = 0;

	//#pragma omp parallel for reduction(+:clasify_ok)
	for(int i = 0; i < n; i++){
		if(leave_one_out) out = i;
		else 	  out = -1;

		if(KNN(Data, T[i], w, out) == T[i][num_atrib-1]){
			clasify_ok+=1;
		}
	}

	return 100.0*clasify_ok/n;
}

double tasaRed(const vector<double> &w){
	int n = w.size();
	int num = 0;

	//#pragma omp parallel for reduction(+:num)
	for(int i = 0; i < n; i++){
		if(w[i] < 0.2)
			num++;
	}

	return 100.0*num/n;
}

double F(const vector< vector <double> > &Data, const vector< vector <double> > &T, const vector<double> &w, bool leave_one_out = true){
	return alpha*tasaClass(Data, T, w, leave_one_out) + (1.0-alpha)*tasaRed(w); 
}


/****************************************************************************************/

void generarVecino(vector<double> &w, int k, double z){
	w[k] += z;

	if(w[k] > 1) w[k] = 1;
	if(w[k] < 0) w[k] = 0;
}

vector<double> SimulatedAnnealing(const vector< vector <double> > &T, int opcion){
	int num_atrib = T[0].size();
	vector<double> w;
	vector<double> best_solution;
	double best_value, actual_value;
	double T0, Tf, Tk, beta;
	int evals = 0, max_eval = 15000;
	double max_vecinos = 10 * num_atrib;
    double max_exitos = 0.1 * max_vecinos;
	int M = 15000/max_vecinos;
	double num_vecinos, num_exitos = 1;

	for(int j = 0; j < num_atrib-1; j++){
		w.push_back(Rand());
	}

	best_solution = w;
	best_value = actual_value = F(T, T,w);

	Tk = T0 = (0.3*best_value)/(-log(0.3));
    Tf = 0.001;

    while(Tk > T0){
    	Tk*=0.001;
    }

    int k = 1;

    beta = (T0 - Tf) / (M*T0*Tf);

	while(Tk > Tf && num_exitos > 0 && evals < max_eval){
		num_exitos = num_vecinos = 0;
		while(num_exitos < max_exitos && num_vecinos < max_vecinos){			
			int i = Randint(0,num_atrib-1);
			double copy = w[i];
			generarVecino(w, i, Normal(0,0.4));
			num_vecinos++;
			evals++;

			double new_value = F(T, T,w);
			double dif = new_value-actual_value;


			if(dif > 0 || Rand() <= exp(dif/(Tk*k))){
				actual_value = new_value;
				num_exitos++;
				if(actual_value > best_value){
					best_solution = w;
					best_value = actual_value;
				}
			}
			else{
				w[i] = copy;
			}
		}
		
		if(opcion == 0)
			Tk = Tk/(1+beta*Tk);
		else
			Tk = Tk*0.95;
	}

	return best_solution;
}

vector<double> SimulatedAnnealing0(const vector< vector <double> > &T){
	return SimulatedAnnealing(T, 0);
}

vector<double> SimulatedAnnealing1(const vector< vector <double> > &T){
	return SimulatedAnnealing(T, 1);
}


/*******************************************************************************************************/


vector<double> BL(const vector< vector <double> > &T, vector<double> w){
	vector<int> ind;
	int n = T.size();
	int num_atrib = T[0].size();
	int iters = 0;
	int nn = 0;
	int nn_top = 20*num_atrib;

	for(int i = 0; i < num_atrib-1; i++){
		ind.push_back(i);
	}

	double value = F(T,T,w);

	while(iters < 1000 && nn < nn_top){
		// Permutamos el vector de indices si ya han sido utilizados todos
		if(iters%(num_atrib-1) == 0)
		   RandShuffle(ind, 0, ind.size());

		int k = ind[iters%(num_atrib-1)];

		double copy = w[k];

		generarVecino(w, k, Normal(0,0.4));

		double new_value = F(T, T,w);
		iters++;
		nn++;

		if(new_value > value){
			value = new_value;
		}
		else{
			w[k] = copy;
		}
	}

	return w;
}


vector<double> ILS(const vector< vector <double> > &T){
	int num_atrib = T[0].size();
	int num_mut = 0.1*num_atrib;
	vector<double> w;
	vector<int> ind;
	double best_value;

	for(int j = 0; j < num_atrib-1; j++){
		w.push_back(Rand());
		ind.push_back(j);
	}

	w = BL(T,w);
	best_value = F(T, T,w);

	for(int i = 0; i < 14; i++){
		vector<double> w_ = w;

		RandShuffle(ind, 0, ind.size());

		for(int j = 0; j < num_mut; j++)
			generarVecino(w, ind[j], Normal(0,0.4));

		w_ = BL(T,w_);

		double new_value = F(T,T,w_);

		if(new_value > best_value){
			best_value = new_value;
			w = w_;
		} 
	}

	return w;
}

/*******************************************************************************************************/

struct chromosome{
	vector<double> w;
	double value;
	chromosome(vector<double> _w, double v){
		w = _w;
		value = v;
	}
	chromosome(){}
};

void initPopulation(vector<chromosome> &population,  const vector< vector <double> > &T, int tam_p, int num_atrib){	

	for(int i = 0; i < tam_p; i++){
		vector<double> w;

		for(int j = 0; j < num_atrib-1; j++){
			w.push_back(Rand());
		}

		population.push_back(chromosome(w, F(T,T,w)));
	}

}

vector<double> DiferentialEvolutionRand(const vector< vector <double> > &T){
	int n = T.size();
	int num_atrib = T[0].size();	
	int tam_p = 50;
	int evals = 0;
	double CR = 0.5;

	vector<chromosome> population;
	vector<int> ind;

	// init population 
	initPopulation(population, T, tam_p, num_atrib);


	for(int i = 0; i < tam_p; i++){
		ind.push_back(i);
	}

	while(evals < 15000){
		vector<chromosome> children;
		for(int i = 0; i < tam_p; i++){
			RandShuffle(ind, 0, ind.size());
			vector<double> father1 = population[ind[0]].w;
			vector<double> father2 = population[ind[1]].w;
			vector<double> father3 = population[ind[2]].w;

			vector<double> child;

			for(int j = 0; j < num_atrib-1; j++){
				if(Rand() < CR){
					double v = father1[j] + 0.5*(father2[j] - father3[j]);
					if(v > 1) v = 1;
					if(v < 0) v = 0;

					child.push_back(v);				
				}
				else{
					child.push_back(population[i].w[j]);
				}
			}
			children.push_back(chromosome(child, F(T,T,child)));
			evals++;
		}

		for(int i = 0; i < tam_p; i++){
			if(children[i].value > population[i].value){
				population[i] = children[i];
			}
		}
	}

	int best = 0;

	for(int i = 1; i < tam_p; i++){
		if(population[i].value > population[best].value)
			best = i;
	}

	return population[best].w;
}


vector<double> DiferentialEvolutionBest(const vector< vector <double> > &T){
	int n = T.size();
	int num_atrib = T[0].size();	
	int k1, k2, tam_p = 50;
	int evals = 0;
	double CR = 0.5;

	vector<chromosome> population;
	vector<int> ind;

	// init population 
	initPopulation(population, T, tam_p, num_atrib);

	int best = 0;

	for(int i = 1; i < tam_p; i++){
		if(population[i].value > population[best].value)
			best = i;
	}

	for(int i = 0; i < tam_p; i++){
		ind.push_back(i);
	}

	while(evals < 15000){
		vector<chromosome> children;
		for(int i = 0; i < tam_p; i++){
			RandShuffle(ind, 0, ind.size());
			vector<double> father1 = population[ind[0]].w;
			vector<double> father2 = population[ind[1]].w;

			vector<double> child;

			for(int j = 0; j < num_atrib-1; j++){
				if(Rand() < CR){
					double v = population[i].w[j] + 0.5*(population[best].w[j]-population[i].w[j]) + 0.5*(father1[j] - father2[j]);
					if(v > 1) v = 1;
					if(v < 0) v = 0;

					child.push_back(v);				
				}
				else{
					child.push_back(population[i].w[j]);
				}
			}
			children.push_back(chromosome(child, F(T,T,child)));
			evals++;
		}

		for(int i = 0; i < tam_p; i++){
			if(children[i].value > population[best].value)
				best = i;

			if(children[i].value > population[i].value){
				population[i] = children[i];
			}
		}
	}

	return population[best].w;
}

/*******************************************************************************************************/

void ExecuteAlgorithm(vector<double> (*algoritmo)(const vector< vector <double> > &), string name){
	struct timespec cgt1,cgt2;

	double tc_mean, tr_mean, func_mean, t_mean;
    double tc, tr, func, ncgt;

	tc_mean = tr_mean = func_mean = t_mean = 0;

	cout << endl << name << endl;

	cout << "tclass / tasa_red / funcion / tiempo " << endl;

	for(int i = 0; i < 5; i++){
		vector< vector <double> > T;

		for(int j = 0; j < 5; j++){
			if(i != j){
				for(int k = 0; k < partitions[j].size(); k++)
					T.push_back(partitions[j][k]);
			}
		}
		clock_gettime(CLOCK_REALTIME,&cgt1);
		vector<double> w = algoritmo(T);
		tc = tasaClass(T, partitions[i], w, false);
		tr = tasaRed(w);
		func = 0.5*tc+0.5*tr;
		clock_gettime(CLOCK_REALTIME,&cgt2);

		ncgt = (double) (cgt2.tv_sec-cgt1.tv_sec)+(double) ((cgt2.tv_nsec-cgt1.tv_nsec)/(1.e+9));

		cout << tc << "  " << tr << "  " << func << "  " << ncgt << endl;

		tc_mean += tc;
		tr_mean += tr;
		func_mean += func;
		t_mean += ncgt;
	}

	cout << "Media" << endl << tc_mean/5.0 << "  " << tr_mean/5.0 << "  " << func_mean/5.0 << "  " << t_mean/5.0 << endl << endl;
}

/**********************************************************************************************************/

int main(int argc, char** argv){
	string option;
	bool ok = false;

	Set_random(17);

	if(argc == 1){
		cout << endl << "Pulse el número que desee ejecutar: " << endl << endl;
		cout << "1: ozone-320.arff" << endl;
		cout << "2: parkinsons.arff" << endl;
		cout << "3: spectf-heart.arff" << endl << endl;
	}
	else{
		option = argv[1];
		ok = true;
	}

	while(!ok){
		cout << "Opción: ";
		getline(cin, option);
		cout << endl;

		if(option[0] > '3' || option[0] < '1' || option.size() != 1)
			cout << "ERROR. Por favor introduce una opción válida." << endl << endl;
		else
			ok = true;		
	}

	string ozone = "./datos/ozone-320.arff";
	string parkinsons  = "./datos/parkinsons.arff";
	string heart = "./datos/spectf-heart.arff";
	
	switch(option[0]){
		case '1': 
			cout << "ozone-320.arff" << endl;
			input(ozone);
		break;
		case '2': 
			cout << "parkinsons.arff" << endl;
			input(parkinsons);
		break;
		case '3':
			cout << "spectf-heart.arff" << endl; 
			input(heart);
		break;
	}

   	Normalize(e);
    createPartitions();

	if(argc == 1){
	    cout << endl << "Elija el algoritmo que desee ejecutar: " << endl << endl;
	    cout << "1: Simulated Annealing" << endl;
	    cout << "2: Búsqueda Local Reiterada" << endl;
	    cout << "3: DiferentialEvoulutionRand" << endl;
	    cout << "4: DiferentialEvoulutionBest" << endl;
	    cout << "5: Simulated Annealing (T = T*0.95)" << endl;

	    ok = false;    	
    }
	else{
		option = argv[2];
		ok = true;
	}

    while(!ok){
		cout << "Opción: ";
		getline(cin, option);

		if(option[0] > '5' || option[0] < '1' || option.size() != 1)
			cout << "ERROR. Por favor introduce una opción válida." << endl << endl;
		else
			ok = true;		
	}

	  alpha = 0.5;

    switch(option[0]){
		case '1': 
			ExecuteAlgorithm(SimulatedAnnealing0, "SimulatedAnnealing");
		break;		
		case '2': 
			ExecuteAlgorithm(ILS, "ILS");
		break;		
		case '3': 
			ExecuteAlgorithm(DiferentialEvolutionRand, "DiferentialEvolutionRand");
		break;
		case '4': 
			ExecuteAlgorithm(DiferentialEvolutionBest, "DiferentialEvolutionBest");
		break;
		case '5': 
			ExecuteAlgorithm(SimulatedAnnealing1, "SimulatedAnnealing (T = T*0.95)");
		break;
	}

}

