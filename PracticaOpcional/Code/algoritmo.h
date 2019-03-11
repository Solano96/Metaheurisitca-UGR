#ifndef __ALGORITMO_H__

#define __ALGORITMO_H__

#include "problemcec2014.h"
#include "problem.h"
//#include "solis.h"
#include "simplex.h"
#include "cmaeshan.h"
#include "random.h"
#include "srandom.h"
#include "domain.h"
#include "localsearch.h"

using namespace realea;

void getInitRandom(Random *random, DomainRealPtr domain, tChromosomeReal &crom) {
   tReal min, max;

   for(unsigned i = 0; i < crom.size(); ++i){
      domain->getValues(i, &min, &max, true);
      crom[i] = random->randreal(min, max);
   }

}


class Algoritmo {
   public:
      unsigned apply(tChromosomeReal &sol, tFitness &fitness, unsigned itera){
         double c1, c2, c3, c4;
         int M, N, L1, L2, EP_max;
         double x_min;
         double x_max;      
         int D = sol.size();
         DomainRealPtr domain = m_problem->getDomain(); 

         int eval = 0;

         switch(D){
            case 10: 
               M = 12; N = 8; L1 = 80; L2 = 1000; EP_max = 5;
               c1 = c2 = 2.05;
               c3 = c4 = 0.5;
            break;
            case 30: 
               M = 20; N = 5; L1 = 280; L2 = 1500; EP_max = 5;
               c1 = c2 = 2.05;
               c3 = c4 = 1.05;
            break;
            case 50:
               M = 20; N = 5; L1 = 250; L2 = 1500; EP_max = 9;
               c1 = c2 = 2.5;
               c3 = c4 = 1.05;
            break;
         }

         

         double L1_div = 1.0/(L1+1.0);
         double L2_div = 0.5/(L2+1.0);

         vector< vector<vector<double> > > x(M, vector< vector<double> >(N, vector<double>(D)));
         vector< vector<vector<double> > > v1(M, vector< vector<double> >(N, vector<double>(D))); 
         vector< vector<vector<double> > > p1(M, vector< vector<double> >(N, vector<double>(D)));

         vector< vector<double> > g(M, vector<double>(D)), v2(M, vector<double>(D)), p2(M, vector<double>(D));
         vector<double> gbest(D);

         vector< vector<double> > p1_value(M, vector<double>(N));
         vector<double> g_value(M), p2_value(M);
         double gbest_value;

         for(int i = 0; i < M; i++){
            for(int j = 0; j < N; j++){
               getInitRandom(m_random, domain, x[i][j]); 
               getInitRandom(m_random, domain, v1[i][j]); 
               getInitRandom(m_random, domain, p1[i][j]);                   
               p1_value[i][j] = m_eval->eval(p1[i][j]);
            }
         }

         for(int i = 0; i < M; i++){
            getInitRandom(m_random, domain, g[i]);
            getInitRandom(m_random, domain, v2[i]);
            getInitRandom(m_random, domain, p2[i]);               
            g_value[i] = m_eval->eval(g[i]);
            p2_value[i] = m_eval->eval(p2[i]);
         }

         getInitRandom(m_random, domain, gbest);
         gbest_value = m_eval->eval(gbest);


         for(int ep = 0; ep < EP_max; ep++){
            // PSO level 1
            for(int i = 0; i < M; i++){
               for(int k = 0; k <= L1; k++){
                  double w1 = 1.0-k*L1_div;

                  for(int j = 0; j < N; j++){
                     double r1 = m_random->rand();
                     double r2 = m_random->rand();

                     for(int d = 0; d < D; d++){
                        v1[i][j][d] = w1*v1[i][j][d] + c1*r1*(p1[i][j][d]-x[i][j][d]) + c2*r2*(g[i][d]-x[i][j][d]);
                        x[i][j][d] = x[i][j][d] + v1[i][j][d];
                     }

                     domain->clip(x[i][j]);
                     double new_value = m_eval->eval(x[i][j]);
                     eval++;

                     if(new_value < p1_value[i][j]){
                        for(int d = 0; d < D; d++)
                           p1[i][j][d] = x[i][j][d];
                        p1_value[i][j] = new_value;

                        if(new_value < g_value[i]){                              
                           for(int d = 0; d < D; d++)
                              g[i][d] = p1[i][j][d];
                           g_value[i] = new_value;

                           if(new_value < gbest_value){
                              for(int d = 0; d < D; d++)
                                 gbest[d] = g[i][d];
                              gbest_value = new_value;
                           }
                        }
                     }
                  }
               }
            }

            // PSO level 2
            vector< vector<double> > y(g);
            
            for(int k = 0; k <= L2; k++){
               double w2 = 1.0-k*1.0*L2_div;
               for(int i = 0; i < M; i++){ 
                  double r3 =  m_random->rand();
                  double r4 =  m_random->rand();

                  for(int d = 0; d < D; d++){
                     v2[i][d] = w2*v2[i][d] + c3*r3*(p2[i][d] - y[i][d]) + c4*r4*(gbest[d]-y[i][d]);
                     y[i][d] = y[i][d] + v2[i][d];
                  }

                  domain->clip(y[i]);
                  double new_value = m_eval->eval(y[i]);
                  eval++;

                  if(new_value < p2_value[i]){
                     for(int d = 0; d < D; d++)
                        p2[i][d] = y[i][d];
                     p2_value[i] = new_value;

                     if(new_value < gbest_value){
                        for(int d = 0; d < D; d++)
                           gbest[d] = p2[i][d];
                        gbest_value = new_value;
                     }
                  }
               }
            }
         }

         if(gbest_value < fitness){
            sol = gbest;
            fitness = gbest_value;
         }

         return eval;
      }

      unsigned apply2(tChromosomeReal &sol, tFitness &fitness, unsigned itera){
         double c1, c2, c3, c4;
         int M, N, L1, L2, EP_max;
         double x_min;
         double x_max;      
         int D = sol.size();
         DomainRealPtr domain = m_problem->getDomain(); 

         int eval = 0;

         switch(D){
            case 10: 
               M = 11; N = 4; L1 = 150; L2 = 800; EP_max = 5;
               c1 = c2 = 2.05;
               c3 = c4 = 1.05;
            break;
            case 30: 
               M = 20; N = 5; L1 = 280; L2 = 1500; EP_max = 5;
               c1 = c2 = 2.05;
               c3 = c4 = 1.05;
            break;
            case 50:
               M = 20; N = 5; L1 = 250; L2 = 1500; EP_max = 9;
               c1 = c2 = 2.05;
               c3 = c4 = 1.05;
            break;
         }

         

         double L1_div = 0.5/(L1+1.0);
         double L2_div = 0.5/(L2+1.0);

         vector< vector<vector<double> > > x(M, vector< vector<double> >(N, vector<double>(D)));
         vector< vector<vector<double> > > v1(M, vector< vector<double> >(N, vector<double>(D))); 
         vector< vector<vector<double> > > p1(M, vector< vector<double> >(N, vector<double>(D)));

         vector< vector<double> > g(M, vector<double>(D)), v2(M, vector<double>(D)), p2(M, vector<double>(D));
         vector<double> gbest(D);

         vector< vector<double> > p1_value(M, vector<double>(N));
         vector<double> g_value(M), p2_value(M);
         double gbest_value;

         for(int i = 0; i < M; i++){
            for(int j = 0; j < N; j++){
               getInitRandom(m_random, domain, x[i][j]); 
               getInitRandom(m_random, domain, v1[i][j]); 
               getInitRandom(m_random, domain, p1[i][j]);                   
               p1_value[i][j] = m_eval->eval(p1[i][j]);
            }
         }

         for(int i = 0; i < M; i++){
            getInitRandom(m_random, domain, g[i]);
            getInitRandom(m_random, domain, v2[i]);
            getInitRandom(m_random, domain, p2[i]);               
            g_value[i] = m_eval->eval(g[i]);
            p2_value[i] = m_eval->eval(p2[i]);
         }

         getInitRandom(m_random, domain, gbest);
         gbest_value = m_eval->eval(gbest);


         for(int ep = 0; ep < EP_max; ep++){
            // PSO level 1
            for(int i = 0; i < M; i++){
               for(int k = 0; k <= L1; k++){

                  double w1 = 1.0-k*L1_div;

                  for(int j = 0; j < N; j++){

                     double r1 = m_random->rand();
                     double r2 = m_random->rand();

                     for(int d = 0; d < D; d++){
                        v1[i][j][d] = w1*v1[i][j][d] + c1*r1*(p1[i][j][d]-x[i][j][d]) 
                        + c2*r2*(g[i][d]-x[i][j][d]);
                        x[i][j][d] = x[i][j][d] + v1[i][j][d];
                     }

                     domain->clip(x[i][j]);
                     double new_value = m_eval->eval(x[i][j]);
                     eval++;

                     if(new_value < p1_value[i][j]){
                        for(int d = 0; d < D; d++)
                           p1[i][j][d] = x[i][j][d];
                        p1_value[i][j] = new_value;

                        if(new_value < g_value[i]){                              
                           for(int d = 0; d < D; d++)
                              g[i][d] = p1[i][j][d];
                           g_value[i] = new_value;

                           if(new_value < gbest_value){
                              for(int d = 0; d < D; d++)
                                 gbest[d] = g[i][d];
                              gbest_value = new_value;
                           }
                        }
                     }
                  }
               }
            }

            // PSO level 2
            vector< vector<double> > y(g);
           
            for(int i = 0;  i < M; i++){
               ILocalSearch *ls;
               ILSParameters *ls_options;
               CMAESHansen *cmaes = new CMAESHansen("cmaesinit.par");    
               cmaes->searchRange(0.1);   
               ls = cmaes;
               ls->setProblem(m_problem);
               ls->setRandom(m_random);
               ls_options = ls->getInitOptions(sol);    
               unsigned evals = ls->apply(ls_options, y[i], p2_value[i], 1000);

               eval += evals;

               for(int d = 0; d < D; d++)
                  p2[i][d] = y[i][d];

               if(p2_value[i] < gbest_value){
                  for(int d = 0; d < D; d++)
                     gbest[d] = p2[i][d];
                  gbest_value = p2_value[i];
               }
            }
         }


         // CMAES sobre la mejor solución

         if(gbest_value < fitness){
            sol = gbest;
            fitness = gbest_value;
         }
         
         ILocalSearch *ls;
         ILSParameters *ls_options;
         //ls = new SimplexDim();
         CMAESHansen *cmaes = new CMAESHansen("cmaesinit.par");    
         cmaes->searchRange(0.1);   
         ls = cmaes;
         ls->setProblem(m_problem);
         ls->setRandom(m_random);
         ls_options = ls->getInitOptions(sol);   
         unsigned evals = ls->apply(ls_options, sol, fitness, 22000);

         eval += evals;

         return eval;
      }

      void setProblem(Problem *problem) {
         m_problem = problem;
         setEval(problem);
      }

      void setRandom(Random *random) {
         m_random = random;
      }

      void setEval(IEval *eval) {
         m_eval = eval;
      }


   protected:
      Random *m_random; /**< The current randomgeneration numbers */
      IEval *m_eval; /** The current evaluation function */
      Problem *m_problem; /**< The current problem */
};


#endif