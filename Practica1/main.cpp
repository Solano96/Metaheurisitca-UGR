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
#include <random>

using namespace std;

vector< vector <double> > e;
vector< vector <vector <double> > > partitions(5);
double alpha = 0.5;


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

int findNearest(const vector< vector <double> > &T, int i, bool _friend){
	double d_min = 99999999;
	int min;	
	int n = T.size();
	int num_atrib = T[0].size();

	for(int k = 0; k < n; k++){
		if(_friend){
		   if(k != i && T[k][num_atrib-1] == T[i][num_atrib-1]){			
				double d = dist(T[i], T[k]);
				if(d < d_min){
					d_min = d;
					min = k;
				}
			}
		}
		else if(T[k][num_atrib-1] != T[i][num_atrib-1]){
			double d = dist(T[i], T[k]);
			if(d < d_min){
				d_min = d;
				min = k;
			}
		}		
	}

	return min;
}

int findNearestEnemy(const vector< vector <double> > &T, int i){
	return findNearest(T, i, false);
}

int findNearestFriend(const vector< vector <double> > &T, int i){
	return findNearest(T, i, true);
}

/*************************************************************************************/
/****************************** ALGORITHM RELIEF *************************************/
/*************************************************************************************/

vector<double> relief(const vector< vector <double> > &T){
	int n = T.size();
	int num_atrib = T[0].size();
	vector<double> w(num_atrib-1, 0);

	for(int i = 0; i < n; i++){
		int near_enemy = findNearestEnemy(T, i);
		int near_friend = findNearestFriend(T, i);

		for(int j = 0; j < num_atrib-1; j++){
			w[j] = w[j] + fabs(T[i][j]-T[near_enemy][j]) - fabs(T[i][j]-T[near_friend][j]);
		}
	}

	double w_max = *max_element(w.begin(), w.end()-1);

	for(int j = 0; j < num_atrib-1; j++){
		if(w[j] < 0) w[j] = 0;
		else         w[j] = 1.0*w[j]/w_max;
	}

	return w;
}

/*************************************************************************************/
/*************************************************************************************/

double tasaClass(const vector< vector <double> > &Data, const vector< vector <double> > &T, const vector<double> &w, bool leave_one_out = true){
	int out, n = T.size();
	int num_atrib = T[0].size();
	int clasify_ok = 0;

	for(int i = 0; i < n; i++){
		if(leave_one_out) out = i;
		else 	  out = -1;

		if(KNN(Data, T[i], w, out) == T[i][num_atrib-1]){
			clasify_ok++;
		}
	}

	return 100.0*clasify_ok/n;
}

double tasaRed(const vector<double> &w){
	int n = w.size();
	int num = 0;

	for(int i = 0; i < n; i++){
		if(w[i] < 0.2)
			num++;
	}

	return 100.0*num/n;
}

double F(const vector< vector <double> > &Data, const vector< vector <double> > &T, const vector<double> &w, bool leave_one_out = true){
	return alpha*tasaClass(Data, T, w, leave_one_out) + (1.0-alpha)*tasaRed(w); 
}

/*************************************************************************************/
/****************************** ALGORITHM BL *****************************************/
/*************************************************************************************/

void generarVecino(vector<double> &w, int k, double z){
	w[k] += z;

	if(w[k] > 1) w[k] = 1;
	if(w[k] < 0) w[k] = 0;
}

vector<double> BL(const vector< vector <double> > &T){
	vector<double> w;
	vector<int> ind;
	int n = T.size();
	int num_atrib = T[0].size();
	int iters = 0;
	int nn = 0;
	int nn_top = 20*num_atrib;

	default_random_engine generator(14);
	normal_distribution<double> normal(0.0, 0.4);
	uniform_real_distribution<double> uniforme(0.0, 1.0);

	for(int i = 0; i < num_atrib-1; i++){
		w.push_back(uniforme(generator));	
		ind.push_back(i);
	}

	double value = F(T,T,w);

	while(iters < 15000 && nn < nn_top){
		// Permutamos el vector de indices si ya han sido utilizados todos
		if(iters%(num_atrib-1) == 0)
		   shuffle(ind.begin(), ind.end(), generator);

		int k = ind[iters%(num_atrib-1)];

		double copy = w[k];

		generarVecino(w, k, normal(generator));

		double new_value = F(T, T,w);
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

	return w;
}



/*************************************************************************************/
/*************************************************************************************/

// EXPERIMENTO


vector<double> ReliefModified(const vector< vector <double> > &T){
	int n = T.size();
	int num_atrib = T[0].size();
	vector<double> w(num_atrib-1, 0);
	vector<int> ind;
	default_random_engine generator(1);

	for(int i = 0; i < n; i++)
		ind.push_back(i);

	shuffle(ind.begin(), ind.end(), generator);

	for(int i = 0; i < n/5; i++){
		int k = ind[i];
		int near_enemy = findNearestEnemy(T, k);
		int near_friend = findNearestFriend(T, k);

		for(int j = 0; j < num_atrib-1; j++){
			w[j] = w[j] + fabs(T[k][j]-T[near_enemy][j]) - fabs(T[k][j]-T[near_friend][j]);
		}
	}

	double w_max = *max_element(w.begin(), w.end()-1);

	for(int j = 0; j < num_atrib-1; j++){
		if(w[j] < 0) w[j] = 0;
		else         w[j] = 1.0*w[j]/w_max;
	}

	return w;
}


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

	cout << "Media" << endl << tc_mean/5.0 << "  " << tr_mean/5.0 << "  " << func_mean/5.0 << "  " << t_mean/5.0 << endl;
}

int main(int argc, char** argv){
	string option;
	bool ok = false;

	cout << endl << "Pulse el número que desee ejecutar: " << endl << endl;
	cout << "1: ozone-320.arff (1-NN, relief, BL)" << endl;
	cout << "2: parkinsons.arff (1-NN, relief, BL)" << endl;
	cout << "3: spectf-heart.arff (1-NN, relief, BL)" << endl << endl;
	cout << "Parte voluntaria: " << endl << endl;
	cout << "4: ozone-320.arff (relief, relief modificado)" << endl;
	cout << "5: parkinsons.arff (relief, relief modificado)" << endl;
	cout << "6: spectf-heart.arff (relief, relief modificado)" << endl << endl;

	cout << "7: parkinsons.arff (BL, BL alfa = 0.2, BL alfa = 1)" << endl << endl;

	cout << "8: genetico" << endl << endl;

	while(!ok){
		cout << "Opción: ";
		getline(cin, option);
		cout << endl;

		if(option[0] > '8' || option[0] < '1' || option.size() != 1)
			cout << "ERROR. Por favor introduce una opción válida." << endl << endl;
		else
			ok = true;		
	}

	string ozone = "./datos/ozone-320.arff";
	string parkinsons  = "./datos/parkinsons.arff";
	string heart = "./datos/spectf-heart.arff";

	if(option[0] == '1' || option[0] == '2' || option[0] == '3'){
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
		// ALGORITMO 1-NN
	    ExecuteAlgorithm(ones, "1-NN");
		// ALGORITMO RELIEF
	    ExecuteAlgorithm(relief, "Relief");
		// ALGORITMO BL		
	    alpha = 0.5;
	    ExecuteAlgorithm(BL, "BL");
	}
	else if(option[0] == '4' || option[0] == '5' || option[0] == '6'){
		switch(option[0]){
			case '4': 
				cout << "ozone-320.arff" << endl;
				input(ozone);
			break;
			case '5': 
				cout << "parkinsons.arff" << endl;
				input(parkinsons);
			break;
			case '6':
				cout << "spectf-heart.arff" << endl; 
				input(heart);
			break;
		}
	   	Normalize(e);
	    createPartitions();
		// ALGORITMO RELIEF
    	ExecuteAlgorithm(relief, "Relief");
    	// Relief Modificado
    	ExecuteAlgorithm(ReliefModified, "Relief Modificado");
	}
	else if(option[0] == '7'){
		cout << "parkinsons.arff" << endl;
		input(parkinsons);	
	   	Normalize(e);
	    createPartitions();	
		// ALGORITMO BL		
	    alpha = 0.5;
	    ExecuteAlgorithm(BL, "BL");
		// ALGORITMO BL alpha = 0.2
	    alpha = 0.2;
	    ExecuteAlgorithm(BL, "BL alfa = 0.2");
	    // ALGORITMO BL alpha = 1
	    alpha = 1;
	    ExecuteAlgorithm(BL, "BL alfa = 1");
	}    
	else if(option[0] == '8'){
		cout << "spectf-heart.arff" << endl;
		input(heart);	
	   	Normalize(e);
	    createPartitions();	
		// ALGORITMO BL		
	    //alpha = 0.5;
	    //ExecuteAlgorithm(BL, "BL");
		// Genetico
	    //alpha = 0.5;
	    //ExecuteAlgorithm(Genetic, "Genetico");
	}   
}



