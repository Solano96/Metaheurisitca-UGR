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
#include <math.h>
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
	do
	 {
	   u1 = Rand();
	   u2 = Rand();
	 }
	while ( u1 <= epsilon );

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

	set<vector<double>> aux;

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

	for(set<vector<double>>::iterator it = aux.begin(); it != aux.end(); it++){
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

struct chromosome{
	vector<double> w;
	double value;
	chromosome(vector<double> _w, double v){
		w = _w;
		value = v;
	}
	chromosome(){}
};

bool comp_chromosome(const chromosome &c1, const chromosome &c2){
	return c1.value > c2.value;
}

/*************************************************************************************/
/******************************** Operadores de cruce ********************************/
/*************************************************************************************/

chromosome crossArithmeticMean(const chromosome &c1, const chromosome &c2, const vector< vector <double> > &T){
	vector<double> w;
	int n = c1.w.size();

	for(int i = 0; i < n; i++){
		w.push_back((c1.w[i]+c2.w[i])/2);
	}

	return chromosome(w, F(T,T,w));
}

double max(double c1, double c2){
	if(c1 >= c2)
		return c1;
	return c2;
}
double min(double c1, double c2){
	if(c1 <= c2)
		return c1;
	return c2;
}

chromosome crossBLX(const chromosome &c1, const chromosome &c2, const vector< vector <double> > &T){
	vector<double> w;
	int n = c1.w.size();

	double cmin, cmax;
	double a = 0.3;
	double I;

	for(int i = 0; i < n; i++){
		cmax = max(c1.w[i], c2.w[i]);
		cmin = min(c1.w[i], c2.w[i]);
		I = cmax-cmin;

		double num = Randfloat(cmin-I*a, cmax+I*a);

		if(num > 1) num = 1;
		if(num < 0) num = 0;
		w.push_back(num);
	}

	return chromosome(w, F(T,T,w));
}

chromosome crossLow(const chromosome &c1, const chromosome &c2, const vector< vector <double> > &T){
	vector<double> w;
	int n = c1.w.size();
	
	for(int i = 0; i < n; i++){
		if(c1.w[i] <= 0.2)
			w.push_back(c1.w[i]);
		else
			w.push_back(c2.w[i]);
	}
	
	return chromosome(w, F(T,T,w));
}

void competition(vector<chromosome> &population, chromosome &child1, chromosome & child2){

	int tam_p = population.size();

	sort(population.begin(), population.end(), comp_chromosome);

	if(child1.value >= child2.value){
		if(population[tam_p-1].value < child1.value){
			population[tam_p-1] = child1;
			if(population[tam_p-2].value < child2.value){
				population[tam_p-2] = child2;
			}
		}
	}
	else if(population[tam_p-1].value < child2.value){
		population[tam_p-1] = child2;
		if(population[tam_p-2].value < child1.value){
			population[tam_p-2] = child1;
		}
	}

}

chromosome binaryTournament(const vector<chromosome> &population){
	chromosome father;
	int k1, k2;	
	int tam_p = population.size();

	k1 = Randint(0, tam_p-1);
	do{k2 = Randint(0, tam_p-1);}while(k1 == k2);

	if(population[k1].value > population[k2].value)
		father = population[k1];
	else
		father = population[k2];

	return father;
}

void initPopulation(vector<chromosome> &population,  const vector< vector <double> > &T, int tam_p, int num_atrib){	

	for(int i = 0; i < tam_p; i++){
		vector<double> w;

		for(int j = 0; j < num_atrib-1; j++){
			w.push_back(Rand());
		}

		population.push_back(chromosome(w, F(T,T,w)));
	}

}

/************************************ GENETICOS *************************************/

vector<double> Stationary(const vector< vector <double> > &T, 
chromosome (*cross)(const chromosome &, const chromosome &, const vector< vector <double> > &)){	
	int n = T.size();
	int num_atrib = T[0].size();	
	int k1, k2, tam_p = 30;
	int evals = 0;

	vector<chromosome> population;

	// init population 
	initPopulation(population, T, tam_p, num_atrib);

	while(evals < 15000){
		chromosome father1, father2, child1, child2;

		father1 = binaryTournament(population);
		father2 = binaryTournament(population);
		child1 = cross(father1, father2, T);
		evals++;

		father1 = binaryTournament(population);
		father2 = binaryTournament(population);
		child2 = cross(father1, father2, T);
		evals++;

		for(int j = 0; j < num_atrib-1; j++){			
			if(Rand() < 0.001){
				child1.w[j] += Normal(0,0.4);
				if(child1.w[j] > 1) child1.w[j] = 1;
				if(child1.w[j] < 0) child1.w[j] = 0;
				child1.value = F(T,T,child1.w);
				evals++;
			}
			if(Rand() < 0.001){
				child2.w[j] += Normal(0,0.4);
				if(child2.w[j] > 1) child2.w[j] = 1;
				if(child2.w[j] < 0) child2.w[j] = 0;
				child2.value = F(T,T,child2.w);
				evals++;
			}
		}
		competition(population, child1, child2);
	}

	int best = 0; 

	for(int i = 1; i < tam_p; i++)
		if(population[i].value > population[best].value)
			best = i;

	return population[best].w;
}

vector<double> StationaryArithmeticMean(const vector< vector <double> > &T){
	return Stationary(T, crossArithmeticMean);
}

vector<double> StationaryBLX(const vector< vector <double> > &T){
	return Stationary(T, crossBLX);
}

vector<double> StationaryLow(const vector< vector <double> > &T){
	return Stationary(T, crossLow);
}

/****************************************************************************************/

void generarVecino(vector<double> &w, int k, double z){
	w[k] += z;

	if(w[k] > 1) w[k] = 1;
	if(w[k] < 0) w[k] = 0;
}

chromosome BL(const vector< vector <double> > &T, chromosome c, int &evals){
	vector<int> ind;
	int n = T.size();
	int num_atrib = T[0].size();
	int iters = 0;
	int nn = 0;
	int nn_top = 20*num_atrib;

	vector<double> w = c.w;
	double value = c.value;

	for(int i = 0; i < num_atrib-1; i++)
		ind.push_back(i);

	while(iters < 2*num_atrib && nn < nn_top){
		// Permutamos el vector de indices si ya han sido utilizados todos
		if(iters%(num_atrib-1) == 0)
			RandShuffle(ind, 0, ind.size());

		int k = ind[iters%(num_atrib-1)];

		double copy = w[k];

		generarVecino(w, k, Normal(0,0.4));

		double new_value = F(T, T,w);
		evals++;
		iters++;
		nn++;

		if(new_value > value){
			nn = 0;
			value = new_value;
		}
		else{
			w[k] = copy;
		}
	}

	return chromosome(w, F(T,T,w));
}

/*******************************************************************************************************/

/********************************************* GENERATIONAL ********************************************/

vector<double> Generational(const vector< vector <double> > &T, 
chromosome (*cross)(const chromosome &, const chromosome &, const vector< vector <double> > &), int memetic = -1){
	int n = T.size();
	int num_atrib = T[0].size();	
	int k1, k2, tam_p;
	int evals = 0;
	int generations = 0;
	vector<chromosome> population;

	if(memetic == -1) tam_p = 30;
	else   tam_p = 10;

	// init population 
	initPopulation(population, T, tam_p, num_atrib);

	int best = 0, worst = 0;

	for(int i = 1; i < tam_p; i++){
		if(population[i].value > population[best].value)
			best = i;
	}

	while(evals < 15000){
		chromosome best_chromosome_old = population[best];

		if(memetic==0 && generations%10 == 0){
			for(int j = 0; j < tam_p; j++)
				population[j] = BL(T,population[j],evals);
		}
		else if(memetic==1 && generations%10 == 0){			
			for(int j = 0; j < tam_p; j++)
				if(Rand() <= 0.1)
					population[j] = BL(T,population[j],evals);
		}
		else if(memetic==2 && generations%10 == 0){
			population[best] = BL(T,population[best],evals);
		}

		for(int j = 0; j < tam_p; j++){
			if(Rand() < 0.7){
				chromosome father1, father2, child;

				father1 = binaryTournament(population);
				father2 = binaryTournament(population);
				child = cross(father1, father2, T);
				evals++;

				for(int k = 0; k < num_atrib-1; k++){
					if(Rand() <= 0.001){
						child.w[k] += Normal(0,0.4);
						if(child.w[k] > 1) child.w[k] = 1;
						if(child.w[k] < 0) child.w[k] = 0;
						child.value = F(T,T,child.w);
					}
				}
				population[j] = child;
			}

			if(population[j].value > population[best].value){
				best = j;
			}
			if(population[j].value < population[worst].value){
				worst = j;
			}
		}

		generations++;
		population[worst] = best_chromosome_old;
	}

	return population[best].w;
}

vector<double> GenerationalArithmeticMean(const vector< vector <double> > &T){
	return Generational(T, crossArithmeticMean);
}

vector<double> GenerationalBLX(const vector< vector <double> > &T){
	return Generational(T, crossBLX);
}

vector<double> GenerationalLow(const vector< vector <double> > &T){
	return Generational(T, crossLow);
}


/************* AM-(10,1.0) every 10 generations BL on all chromosomes *************/ 

vector<double> MemeticArithmeticMean_10_1(const vector< vector <double> > &T){
	return Generational(T, crossArithmeticMean, 0);
}

vector<double> MemeticBLX_10_1(const vector< vector <double> > &T){
	return Generational(T, crossBLX, 0);
}

vector<double> MemeticLow_10_1(const vector< vector <double> > &T){
	return Generational(T, crossLow, 0);
}


/***** AM-(10,0.1) every 10 generations BL with 0.1 probability on chromosomes *****/

vector<double> MemeticArithmeticMean_10_01(const vector< vector <double> > &T){
	return Generational(T, crossArithmeticMean, 1);
}

vector<double> MemeticBLX_10_01(const vector< vector <double> > &T){
	return Generational(T, crossBLX, 1);
}

vector<double> MemeticLow_10_01(const vector< vector <double> > &T){
	return Generational(T, crossLow, 1);
}


/********* AM-(10,0.1best) every 10 generations BL on 0.1N best chromosomes *********/

vector<double> MemeticArithmeticMean_10_01_best(const vector< vector <double> > &T){
	return Generational(T, crossArithmeticMean, 2);
}

vector<double> MemeticBLX_10_01_best(const vector< vector <double> > &T){
	return Generational(T, crossBLX, 2);
}

vector<double> MemeticLow_10_01_best(const vector< vector <double> > &T){
	return Generational(T, crossLow, 2);
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
	    cout << "1: Estacionario Media Aritmética" << endl;
	    cout << "2: Estacionario BLX" << endl;
	    cout << "3: Generacional Media Aritmética" << endl;
	    cout << "4: Generacional BLX" << endl;
	    cout << "5: Memético Media Aritmética" << endl;
	    cout << "6: Memético BLX" << endl << endl;
	    cout << "Voluntario:" << endl <<endl;
	    cout << "7: Estacionario Low" << endl;
	    cout << "8: Generacional Low" << endl;
	    cout << "9: Memetico Low" << endl << endl;
	    ok = false;    	
    }
	else{
		option = argv[2];
		ok = true;
	}

    while(!ok){
		cout << "Opción: ";
		getline(cin, option);

		if(option[0] > '9' || option[0] < '1' || option.size() != 1)
			cout << "ERROR. Por favor introduce una opción válida." << endl << endl;
		else
			ok = true;		
	}

    alpha = 0.5;

    switch(option[0]){
		case '1': 
			ExecuteAlgorithm(StationaryArithmeticMean, "Estacionario Media Aritmética");
		break;
		case '2': 
			ExecuteAlgorithm(StationaryBLX, "Estacionario BLX");
		break;
		case '3':
			ExecuteAlgorithm(GenerationalArithmeticMean, "Generacional Media Aritmética");
		break;
		case '4': 
			ExecuteAlgorithm(GenerationalBLX, "Generacional BLX");
		break;
		case '7': 
			ExecuteAlgorithm(StationaryLow, "Estacionario Low");
		break;
		case '8': 
			ExecuteAlgorithm(GenerationalLow, "Generacional Low");
		break;
	}

	string optionBL;

	if(option[0] == '5' || option[0] == '6' || option[0] == '9'){
		if(argc == 1){
			cout << endl << "Elija como quiere realizar el BL: " << endl << endl;
		    cout << "1: Cada 10 generaciones sobre todos los cromosomas" << endl;
		    cout << "2: Cada 10 generaciones con probabilidad de 0.1 por cromosoma" << endl;
		    cout << "3: Cada 10 generaciones sobre el 0.1N mejores" << endl << endl;
		    ok = false;
		}
		else{
			optionBL = argv[3];
			ok = true;
		}

	    while(!ok){
			cout << "Opción: ";
			getline(cin, optionBL);

			if(optionBL[0] > '3' || optionBL[0] < '1' || optionBL.size() != 1)
				cout << "ERROR. Por favor introduce una opción válida." << endl << endl;
			else
				ok = true;		
		}

		if(option[0] == '5'){
			switch(optionBL[0]){
				case '1': 
					ExecuteAlgorithm(MemeticArithmeticMean_10_1, "Memetico Media Aritmética AM-(10,1.0)");
				break;
				case '2': 
					ExecuteAlgorithm(MemeticArithmeticMean_10_01, "Memetico Media Aritmética AM-(10,0.1)");
				break;
				case '3':
					ExecuteAlgorithm(MemeticArithmeticMean_10_01_best, "Memetico Media Aritmética AM-(10,0.1mej)");
				break;
			}
		}
		else if(option[0] == '6'){
			switch(optionBL[0]){
				case '1': 
					ExecuteAlgorithm(MemeticBLX_10_1, "Memetico BLX AM-(10,1.0)");
				break;
				case '2': 
					ExecuteAlgorithm(MemeticBLX_10_01, "Memetico BLX AM-(10,0.1)");
				break;
				case '3':
					ExecuteAlgorithm(MemeticBLX_10_01_best, "Memetico BLX AM-(10,0.1mej)");
				break;
			}
		}
		else{
			switch(optionBL[0]){
				case '1': 
					ExecuteAlgorithm(MemeticLow_10_1, "Memetico Low AM-(10,1.0)");
				break;
				case '2': 
					ExecuteAlgorithm(MemeticLow_10_01, "Memetico Low AM-(10,0.1)");
				break;
				case '3':
					ExecuteAlgorithm(MemeticLow_10_01_best, "Memetico Low AM-(10,0.1mej)");
				break;
			}
		}
	}
	
}

