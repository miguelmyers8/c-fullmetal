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

using namespace std;



Tensor::Tensor(){};

Tensor::Tensor(xt::xarray<double> a, bool b){
    data = a; 
    requires_grad = b;
    shape = data.shape();
}

// Class to hold child of parents
Dependancies::Dependancies(Tensor a, std::function<xt::xarray<double>( xt::xarray<double> )> b){
    tenor = a;
    grad_fn = b;
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
        if(requires_grad){
            // grad function
            auto add_grad = [&] (xt::xarray<double> y) {
                return out + y;
            };
        };
        return Tensor(out);
    };



std::ostream& operator<<(std::ostream &strm, const Tensor &a) {
    return strm << "Tensor( " << a.data << ", "<< std::boolalpha << a.requires_grad <<" )";}





