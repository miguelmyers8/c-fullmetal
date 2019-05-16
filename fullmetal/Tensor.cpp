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
using namespace std;


Tensor::Tensor(xt::xarray<double> a, bool b){
    data = a;
    requires_grad = b;
    shape = data.shape();
}

Tensor add(Tensor t1, Tensor t2){
    auto _data = t1.data + t2.data;
    Tensor tensor_add(_data);
    return tensor_add;
    };

std::ostream& operator<<(std::ostream &strm, const Tensor &a) {
    return strm << "Tensor(" << a.data << ", "<< std::boolalpha << a.requires_grad <<")";}




