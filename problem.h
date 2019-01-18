//
// Created by liuxi on 2019/1/2.
//

#ifndef NUMOPT3_PROBLEM_H
#define NUMOPT3_PROBLEM_H

#include "matrix.h"
#include "matrix_operation.h"
#include "hyperdual.h"
#include <functional>

struct Cost {
    virtual Matrix<double>& get_variables() = 0;

    virtual double evaluate() { return evaluate(get_variables()); }
    virtual void gradient(Matrix<double>& grad) { return gradient(get_variables(),grad); }
    virtual void hessian(Matrix<double>& hess) { return hessian(get_variables(),hess); }

    virtual int variable_count() = 0;
    virtual double evaluate(const Matrix<double>&vars) = 0;
    virtual void gradient(const Matrix<double>&vars, Matrix<double>& grad) = 0;
    virtual void hessian(const Matrix<double>&vars, Matrix<double>& hess) = 0;
};

template <template <typename> class COST_FUNCTOR>
struct CostGH: Cost {
    Matrix<double> variables;
    COST_FUNCTOR<double> eval_functor;
    COST_FUNCTOR<hyperdual> grad_functor;
    COST_FUNCTOR<hyperdual> hess_functor;

    double evaluate(const Matrix<double>&vars) {
        //static int i = 0;
        double r = eval_functor(vars);
        //cout<<i++<<": "<<r<<endl;
        return r;
    }

    void gradient(const Matrix<double>&vars, Matrix<double>& grad) {
        assert(vars.cols()==1);
        int n = vars.rows();
        Matrix<hyperdual> m(n,1,hyperdual());

        for(int i=0;i<n;i++)
            m(i,0).f0 = vars(i,0);

        for(int i=0;i<n;) {
            if (i+1<n) {
                m(i,0).f1 = 1.0f;
                m(i+1,0).f2 = 1.0f;
                hyperdual h = grad_functor(m);
                grad(i,0) = h.eps1();
                grad(i+1,0) = h.eps2();
                m(i,0).f1 = 0.0f;
                m(i+1,0).f2 = 0.0f;
                i+=2;
            } else {
                m(i,0).f1 = 1.0f;
                hyperdual h = grad_functor(m);
                grad(i,0) = h.eps1();
                m(i,0).f1 = 0.0f;
                i+=1;
            }
        }
    }

    void hessian(const Matrix<double>&vars, Matrix<double>& hess) {
        assert(vars.cols()==1);
        int n = vars.rows();
        Matrix<hyperdual> m(n,1,hyperdual());

        for(int i=0;i<n;i++)
            m(i,0).f0 = vars(i,0);

        for(int r=0;r<n;r++) {
            for(int c=0;c<=r;c++) {
                m(r,0).f1 = 1.0f;
                m(c,0).f2 = 1.0f;

                hyperdual h = hess_functor(m);
                hess(r,c) = hess(c,r) = h.eps1eps2();

                m(r,0).f1 = 0.0f;
                m(c,0).f2 = 0.0f;
            }
        }
    }

    Matrix<double> &get_variables() override {
        return variables;
    }

    CostGH(int n) : variables(n,1,0.0) {}

    int variable_count() override {
        return variables.rows();
    }
};


#endif //NUMOPT3_PROBLEM_H
