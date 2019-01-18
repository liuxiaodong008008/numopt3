#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#include <iostream>
#include "hyperdual.h"
#include "matrix.h"
#include "matrix_operation.h"
#include <algorithm>
#include <iomanip>
#include "problem.h"
#include "optimizer.h"
#include "time.h"
using namespace std;

//auto y = vars(1);
//return -exp(-(x-T(3.1))*(x-T(3.1)))+0.1*(x-T(0.4))*(x-T(0.4));
//return exp(-(x-T(3.1))*(x-T(3.1)))+x*y+(y-x)*(y-x);
//return x*x + y*x/2 + 10*sin((x+y)*(x+y)/30);

template <typename T>
struct MyFunctor {
    virtual T operator()(const Matrix<T>& vars) const {
        auto x = vars(0);
        return (x-T(2.2))*x;
    }
};


int main() {

    {
        time_t start_time, end_time;
        start_time = clock();

        CostGH<MyFunctor> cost(1);
        BacktrackingStepSolver stepSolver(1, 0.5, 1e-4);
        cost.variables(0) = 3.0f;
        DFPRank1QuasiNewtonMethod opt(cost,stepSolver,1e-10);
        opt.minimize();

        end_time = clock();
        cout<<cost.get_variables()<<endl;
        cout<<"time used: "<<setprecision(9)<<(end_time-start_time+0.0)/CLOCKS_PER_SEC<<endl;
    }


    {
        time_t start_time, end_time;
        start_time = clock();

        CostGH<MyFunctor> cost(1);
        BacktrackingStepSolver stepSolver(1, 0.5, 1e-4);
        cost.variables(0) = 3.0f;
        BFGSRank2QuasiNewtonMethod opt(cost,stepSolver,1e-10);
        opt.minimize();

        end_time = clock();
        cout<<cost.get_variables()<<endl;
        cout<<"time used: "<<setprecision(9)<<(end_time-start_time+0.0)/CLOCKS_PER_SEC<<endl;
    }

    {
        time_t start_time, end_time;
        start_time = clock();

        CostGH<MyFunctor> cost(1);
        BacktrackingStepSolver stepSolver(1, 0.5, 1e-4);
        cost.variables(0) = 3.0f;
        SR1QuasiNewtonMethod opt(cost,stepSolver,1e-10);
        opt.minimize();

        end_time = clock();
        cout<<cost.get_variables()<<endl;
        cout<<"time used: "<<setprecision(9)<<(end_time-start_time+0.0)/CLOCKS_PER_SEC<<endl;
    }


    {
        time_t start_time, end_time;
        start_time = clock();

        CostGH<MyFunctor> cost(1);
        BacktrackingStepSolver stepSolver(1, 0.5, 1e-4);
        cost.variables(0) = 3.0f;
        NewtonMethod opt(cost,stepSolver,1e-9);
        opt.minimize();

        end_time = clock();
        cout<<cost.get_variables()<<endl;
        cout<<"time used: "<<setprecision(9)<<(end_time-start_time+0.0)/CLOCKS_PER_SEC<<endl;
    }

    {
        time_t start_time, end_time;
        start_time = clock();

        CostGH<MyFunctor> cost(1);
        cost.variables(0) = 3.0f;
        ConjugateGradient opt(cost,1e-5);
        opt.minimize();

        end_time = clock();
        cout<<cost.get_variables()<<endl;
        cout<<"time used: "<<setprecision(9)<<(end_time-start_time+0.0)/CLOCKS_PER_SEC<<endl;
    }


    _CrtDumpMemoryLeaks();
}