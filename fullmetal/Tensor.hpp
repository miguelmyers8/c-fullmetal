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
#include <vector>
#include <optional>
#include <memory>
using namespace std;


class Tensor;
class Dependancies;


class Tensor {
    
    public:
    
        xt::xarray<double> data;
        xt::svector<size_t> shape;
        std::vector<Dependancies> depend_on;
        shared_ptr<Tensor>grad;
        bool requires_grad;
    
        Tensor();

        Tensor(xt::xarray<double> a, bool b = false, std::vector<Dependancies> d = {} ); // constructor
    
        ~Tensor();
    
        friend std::ostream& operator<<(std::ostream &strm, const Tensor &a); // stream overload
    
        friend Tensor operator+(const Tensor& a, const Tensor& b); //Tensor + Tensor
    
        friend Tensor operator+(const int& a, const Tensor& b); // int + Tensor
    
        friend Tensor operator+(const Tensor& a, const int& b); // Tensor + int
    
        friend Tensor add(Tensor t1, Tensor t2); // add funtion will be called from operator+

        void zero_grad();
};


class Dependancies{
    
    public:
    
        Tensor tenor;
    
        string name;
    
        std::function<xt::xarray<double>( xt::xarray<double> )> grad_fn;
    
        Dependancies();
    
        Dependancies(Tensor a,  std::function<xt::xarray<double>( xt::xarray<double> )> b, string c);
    
        ~Dependancies();
    
        friend std::ostream& operator<<(std::ostream &strm, const Dependancies &a); // stream overload
};



#endif /* Tensor_hpp */
