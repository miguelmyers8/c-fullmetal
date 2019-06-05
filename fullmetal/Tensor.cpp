//
//  Tensor.cpp
//  fullmetal
//
//  Created by miguel myers on 5/15/19.
//  Copyright © 2019 miguel myers. All rights reserved.
//

#include "Tensor.hpp"
#include <iostream>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <boost/variant/variant.hpp>
#include <memory>
using namespace std;

Tensor::Tensor(){};
Dependancies::Dependancies(){};



Tensor::Tensor(xt::xarray<double> a, bool b, std::vector<Dependancies> d){
    data = a; 
    requires_grad = b;
    depend_on = d;
    shape = data.shape();
    grad = nullptr;
    if (this->requires_grad){this->zero_grad();};
    
};
Tensor::~Tensor(){};



Dependancies::Dependancies(Tensor a, std::function<xt::xarray<double>( xt::xarray<double> )> b, string c){
    tenor = a;
    grad_fn = b;
    name = c;
};
Dependancies::~Dependancies(){};


void Tensor::zero_grad(){
    auto i = full_like(this->data, 0.);
    shared_ptr<Tensor> k = make_shared<Tensor>(i);
    this->grad = k;
};

// add
    Tensor operator+(const Tensor & a, const Tensor& b){
        return add(a,b);
    };

    Tensor operator+(const int & a, const Tensor& b){
       Tensor _a = Tensor(a);
        return add(_a,b);
    };

    Tensor operator+(const Tensor & a, const int& b){
        Tensor _b = Tensor(b);
        return add(a,_b);
    };

    Tensor add(Tensor t1, Tensor t2){
        auto out = t1.data + t2.data;
        bool requires_grad = t1.requires_grad || t2.requires_grad;
        std::vector<Dependancies> depends_on = {};
        
        if(t1.requires_grad){
            // grad function: needs finishing
            auto add_grad_1 = [&] (xt::xarray<double> y) {
                xt::xarray<double> grad_out = out + y;
                return grad_out;
            };
            depends_on.push_back(Dependancies(t1,add_grad_1,"add.<add_grad_1>"));
        };
        return Tensor(out,requires_grad,depends_on);
    };



std::ostream& operator<<(std::ostream &strm, const Tensor &a) {
    return strm << "Tensor("<< a.data << ", requires_grad= "<< std::boolalpha << a.requires_grad <<")"<<endl;}

std::ostream& operator<<(std::ostream &strm, const Dependancies &a) {
    return strm << "Dependancies( Tensor= "<<"Tensor("<< a.tenor.data << "), grad_fn= " <<a.name<<")"<<endl;}



