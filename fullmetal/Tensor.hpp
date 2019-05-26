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
#include <boost/variant/variant.hpp>
#include <tuple>

using namespace std;






class Tensor {
    
    public:
    
        xt::xarray<double> data;
        xt::svector<size_t> shape;
        xt::xarray<double> grad;
        bool requires_grad;
    
        Tensor();

        Tensor(xt::xarray<double> a, bool b = false); // constructor
    
        friend std::ostream& operator<<(std::ostream &strm, const Tensor &a); // stream overload
    
        friend Tensor operator+(const Tensor& a, const Tensor& b); //Tensor + Tensor
    
        friend Tensor operator+(const int& a, const Tensor& b); // int + Tensor
    
        friend Tensor operator+(const Tensor& a, const int& b); // Tensor + int
    
        friend Tensor add(Tensor t1, Tensor t2); // add funtion will be called from operator+

    
};


class Dependancies{
    
    public:
    
        Tensor tenor;
    
        std::function<xt::xarray<double>( xt::xarray<double> )> grad_fn;
    
        Dependancies(Tensor a,  std::function<xt::xarray<double>( xt::xarray<double> )> b);
};



#endif /* Tensor_hpp */
