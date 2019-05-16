//
//  Tensor.hpp
//  fullmetal
//
//  Created by miguel myers on 5/15/19.
//  Copyright © 2019 miguel myers. All rights reserved.
//

#ifndef Tensor_hpp
#define Tensor_hpp

#include <stdio.h>

#include <iostream>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

using namespace std;


class Tensor {
    
public:
    xt::xarray<double> data;
    xt::svector<size_t> shape;
    xt::xarray<double> grad;
    bool requires_grad;
    
    Tensor(xt::xarray<double> a, bool b=false);
    
    friend std::ostream& operator<<(std::ostream &strm, const Tensor &a);
    
    friend  Tensor add(Tensor t1, Tensor t2);

   
};

std::ostream& operator<<(std::ostream &strm, const Tensor &a);
Tensor add(Tensor t1, Tensor t2);


#endif /* Tensor_hpp */
