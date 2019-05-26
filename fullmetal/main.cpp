//
//  main.cpp
//  fullmetal
//
//  Created by miguel myers on 5/6/19.
//  Copyright © 2019 miguel myers. All rights reserved.
//

#include <iostream>
#include "Tensor.hpp"
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <typeinfo>
#include <tuple>
#include <string>



int main(int argc, const char * argv[]) {
    
    auto aa = xt::random::randn<double> ({2, 2});
    auto bb = xt::random::randn<double> ({2, 2});
    
    xt::xarray<double> k = {{ 11, 12, 13 }};
    xt::xarray<double> l = {  1,  2,  3 };
    auto v = k+l;
    
    auto add_grad = [&] (xt::xarray<double> y) {
        xt::xarray<double> j  = aa + y;
        return j;
    };
 
    
    Tensor t1(k);
    
    Dependancies b(t1,add_grad);
    
    cout << b.grad_fn(bb);
        return 0;
}
