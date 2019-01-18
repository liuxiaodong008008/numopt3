//
// Created by liuxi on 2019/1/3.
//

#ifndef NUMOPT3_OPTIMIZER_H
#define NUMOPT3_OPTIMIZER_H

#include "problem.h"

struct Optimizer{
    virtual void minimize() = 0;
};

struct StepSolver{
    virtual double get_step(Cost & cost, const Matrix<double>& direct) = 0;
};

struct BacktrackingStepSolver:StepSolver {
    double get_step(Cost &cost, const Matrix<double> &direct) override {
        double alpha = this->alpha;
        double rho = this->rho;
        double c = this->c;

        Matrix<double> vars = cost.get_variables();
        Matrix<double> d = direct;
        Matrix<double> grad(direct.rows(),1);
        cost.gradient(grad);
        double descent_factor = matmul(grad.t(),direct)(0);

        auto cur_cost = cost.evaluate();
        int i=0;
        while(true){
            i++;
            auto new_cost = cost.evaluate(vars+alpha*direct);
            if(new_cost<=cur_cost+c*alpha*descent_factor) break;
            else alpha = alpha*rho;
        }
        //cout<<i<<endl;
        return alpha;
    }

    double alpha;
    double rho;
    double c;
    BacktrackingStepSolver(double alpha, double rho, double c)
        : alpha(alpha), rho(rho), c(c){}
};

struct InterpolationStepSolver:StepSolver {
    double get_step(Cost &cost, const Matrix<double> &direct) override {
        double alpha = this->alpha;
        double c = this->c;

        Matrix<double> vars = cost.get_variables();
        Matrix<double> d = direct;
        Matrix<double> grad(direct.rows(),1);
        cost.gradient(grad);

        auto cur_cost = cost.evaluate();
        double phi0p = matmul(grad.t(),direct)(0);
        double phi0 = cur_cost;
        double descent_factor = phi0p;

        int i=0;
        while (true)
        {
            i++;
            auto new_cost = cost.evaluate(vars+alpha*direct);
            if(new_cost<=cur_cost+c*alpha*descent_factor) break;
            else {
                alpha = - (phi0p*alpha*alpha)/(2*(new_cost-phi0-phi0p*alpha));
            }
        }
        cout<<i<<endl;
        return alpha;
    }

    double alpha;
    double c;
    InterpolationStepSolver(double alpha, double c)
            : alpha(alpha), c(c){}
};

struct ConstStepSolver:StepSolver {
    double get_step(Cost &cost, const Matrix<double> &direct) override {
        return step;
    }

    double step;

    ConstStepSolver(double step) : step(step) {}
};

struct GradientDescent:Optimizer {
    Cost & cost;
    StepSolver & stepSolver;
    double epsilon;

    GradientDescent(Cost &cost, StepSolver &stepSolver, double epsilon=1e-8) : cost(cost), stepSolver(stepSolver),
                                                                         epsilon(epsilon) {}

    virtual ~GradientDescent() {
//        if(cost != nullptr) {
//            delete cost;
//            cost = nullptr;
//        }
//
//        if(stepSolver != nullptr) {
//            delete stepSolver;
//            stepSolver = nullptr;
//        }
    }

    void minimize() override {
        Matrix<double> grad(cost.variable_count(),1,0.0);
        int i=0;
        double last_cost = cost.evaluate();
        while (true) {
            cost.gradient(grad);
            auto direct = -grad;
            auto step = stepSolver.get_step(cost,-grad);

            auto xx = -step*direct;
            cout<<cost.get_variables()(0)<<","<<cost.get_variables()(1)<<","<<xx(0)<<","<<xx(1)<<endl;

            cost.get_variables()+=step*(-grad);
            double cur_cost = cost.evaluate();
            if(last_cost-cur_cost<epsilon) {
                break;
            }
            last_cost = cur_cost;
        }
    }
};

//struct NewtonMethod:Optimizer {
//    Cost & cost;
//    StepSolver & stepSolver;
//    double epsilon;
//
//    NewtonMethod(Cost &cost, StepSolver &stepSolver, double epsilon=1e-8)
//        : cost(cost), stepSolver(stepSolver), epsilon(epsilon) {}
//    virtual ~NewtonMethod() {}
//
//    void minimize() override {
//        Matrix<double> grad(cost.variable_count(),1,0.0);
//        Matrix<double> hess(cost.variable_count(),cost.variable_count(),0.0);
//        int i=0;
//        double last_cost = cost.evaluate();
//        while (true) {
//            cost.gradient(grad);
//            cost.hessian(hess);
//            auto direct = -matmul(inverse(hess),grad);
//            auto step = stepSolver.get_step(cost,direct);
//            auto xx = step*direct;
//            cout<<cost.get_variables()(0)<<","<<cost.get_variables()(1)<<","<<xx(0)<<","<<xx(1)<<endl;
//            cost.get_variables()+=step*direct;
//            double cur_cost = cost.evaluate();
//            if(abs(last_cost-cur_cost)<epsilon) {
//                break;
//            }
//            last_cost = cur_cost;
//        }
//    }
//};

struct MatrixUpdater{
    virtual void updateMatrix(const Matrix<double>&sk,const Matrix<double>&yk,Matrix<double>&mat) const = 0;
};

struct DFPRank1MatrixUpadter: MatrixUpdater {
    virtual void updateMatrix(const Matrix<double> &sk, const Matrix<double> &yk, Matrix<double> &mat) const override {
        auto &Hk = mat;
        auto t = matmul(Hk,yk);
        Hk = Hk-matmul(t,t.t())/matmul(yk.t(),t)+matmul(sk,sk.t())/matmul(yk.t(),sk);
    }
};

struct NewtonMethod:Optimizer {
    Cost & cost;
    StepSolver & stepSolver;
    double epsilon;

    NewtonMethod(Cost &cost, StepSolver &stepSolver, double epsilon)
            : cost(cost),
              stepSolver(stepSolver),
              epsilon(epsilon) {}

    void minimize() override {
        Matrix<double> gradk(cost.variable_count(),1,0.0);
        Matrix<double> gradk1(cost.variable_count(),1,0.0);
        Matrix<double> sk(cost.variable_count(),1,0.0);
        Matrix<double> Hk(cost.variable_count(),cost.variable_count(),0.0);

        cost.gradient(gradk);
        cost.hessian(Hk);
        Hk = inverse(Hk);

        while (true) {

            //cout<<cost.get_variables()<<endl;

            auto direct = -matmul(Hk,gradk);
            auto step = stepSolver.get_step(cost,direct);
            sk = step*direct;
            cost.get_variables()+=sk;

            cost.gradient(gradk1);
            auto m = magnitude(gradk1);

            //cout<<"|grad|="<<m<<endl;
            if(m<epsilon) break;

            cost.hessian(Hk);
            Hk = inverse(Hk);
            gradk = gradk1;
        }
    }
};


struct QuasiNewtonMethod:Optimizer {
    Cost & cost;
    StepSolver & stepSolver;
    double epsilon;
    MatrixUpdater & matrixUpdater;

    QuasiNewtonMethod(Cost &cost, StepSolver &stepSolver, double epsilon, MatrixUpdater &matrixUpdater)
    : cost(cost),
      stepSolver(stepSolver),
      epsilon(epsilon),
      matrixUpdater(matrixUpdater) {}

    void minimize() override {
        Matrix<double> gradk(cost.variable_count(),1,0.0);
        Matrix<double> gradk1(cost.variable_count(),1,0.0);
        Matrix<double> sk(cost.variable_count(),1,0.0);
        Matrix<double> yk(cost.variable_count(),1,0.0);
        Matrix<double> Hk(cost.variable_count(),cost.variable_count(),0.0);

        cost.gradient(gradk);
        cost.hessian(Hk);
        Hk = inverse(Hk);

        while (true) {
            auto direct = -matmul(Hk,gradk);
            auto step = stepSolver.get_step(cost,direct);
            sk = step*direct;
            cost.get_variables()+=sk;

            cost.gradient(gradk1);
            auto m = magnitude(gradk1);
            if(m<epsilon) break;

            yk = gradk1-gradk;
            matrixUpdater.updateMatrix(sk,yk,Hk);
            gradk = gradk1;
        }
    }
};

struct DFPRank1QuasiNewtonMethod:Optimizer {
    QuasiNewtonMethod qnm;
    DFPRank1MatrixUpadter dfpm;

    DFPRank1QuasiNewtonMethod(Cost &cost, StepSolver &stepSolver, double epsilon)
    : dfpm(),qnm(cost,stepSolver,epsilon,this->dfpm) {}
    void minimize() override {
        qnm.minimize();
    }
};

struct BFGSRank2MatrixUpadter: MatrixUpdater {
    virtual void updateMatrix(const Matrix<double> &sk, const Matrix<double> &yk, Matrix<double> &mat) const override {
        auto &Hk = mat;
        auto rhok = 1/matmul(yk.t(),sk)(0);
        auto t = Matrix<double>::eye(Hk.rows()) - rhok*matmul(sk,yk.t());
        Hk = matmul(matmul(t,Hk),t.t())+rhok*matmul(sk,sk.t());
    }
};


struct BFGSRank2QuasiNewtonMethod:Optimizer {
    QuasiNewtonMethod qnm;
    BFGSRank2MatrixUpadter bfgsr2;

    BFGSRank2QuasiNewtonMethod(Cost &cost, StepSolver &stepSolver, double epsilon)
            : bfgsr2(),qnm(cost,stepSolver,epsilon,this->bfgsr2) {}
    void minimize() override {
        qnm.minimize();
    }
};


struct SR1MatrixUpadter: MatrixUpdater {
    virtual void updateMatrix(const Matrix<double> &sk, const Matrix<double> &yk, Matrix<double> &mat) const override {
        auto &Hk = mat;
        auto t = sk-matmul(Hk,yk);
        Hk = Hk+matmul(t,t.t())/(t.t(),yk);
    }
};


struct SR1QuasiNewtonMethod:Optimizer {
    QuasiNewtonMethod qnm;
    SR1MatrixUpadter sr1;

    SR1QuasiNewtonMethod(Cost &cost, StepSolver &stepSolver, double epsilon)
            : sr1(),qnm(cost,stepSolver,epsilon,this->sr1) {}
    void minimize() override {
        qnm.minimize();
    }
};

//
//struct ConjugateGradient:Optimizer {
//    Cost & cost;
//    double epsilon;
//
//    ConjugateGradient(Cost &cost, double epsilon) : cost(cost), epsilon(epsilon) {}
//
//    void minimize() override {
//        Matrix<double> gradk(cost.variable_count(),1,0.0);
//        Matrix<double> gradk1(cost.variable_count(),1,0.0);
//        Matrix<double> hess(cost.variable_count(),cost.variable_count(),0.0);
//        Matrix<double> rk(cost.variable_count(),1,0.0);
//        Matrix<double> pk(cost.variable_count(),1,0.0);
//
//        cost.gradient(gradk);
//        cost.hessian(hess);
//
//        rk = matmul(hess,cost.get_variables())-gradk;
//        pk = -rk;
//        double rktrk = matmul(rk.t(),rk)(0);
//
//        while (true) {
//            cout<<"x"<<endl;
//            //cout<<cost.get_variables()<<endl;
//            auto alpha = -matmul(rk.t(),pk)(0)/matmul(pk.t(),matmul(hess,pk))(0);
//            cost.get_variables()+=alpha*pk;
//            rk+=matmul(hess,pk)+gradk;
//
//            double beta = matmul(matmul(rk.t(),hess),pk)(0)/matmul(matmul(pk.t(),hess),pk)(0);
//            pk=-rk+beta*pk;
//
//            double m = magnitude(rk);
//            cout<<"|rk|="<<m<<endl;
//            if(m<epsilon) break;
//        }
//    }
//};



#endif //NUMOPT3_OPTIMIZER_H
