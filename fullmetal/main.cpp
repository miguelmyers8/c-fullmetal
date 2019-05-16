//
//  main.cpp
//  fullmetal
//
//  Created by miguel myers on 5/6/19.
//  Copyright © 2019 miguel myers. All rights reserved.
//

#include <iostream>
#include "Tensor.h"
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <typeinfo>
#include <tuple>


int main(int argc, const char * argv[]) {
    // insert code here...
    auto a = xt::random::randn<double> ({2, 2});
    auto b = xt::random::randn<double> ({2, 2});
    
    xt::xarray<double> k = {{ 11, 12, 13 }};
    xt::xarray<double> l = {  1,  2,  3 };
   
    Tensor t1(a);
    xt::svector<size_t> x = t1.shape;
    cout<<x[0]<<endl;
    //t1.test();
    //cout<<t1.requires_grad<<endl;
    
    //xt::svector<size_t> y = k.shape();
    
    //for (auto& el : y) {
        //std::cout << el <<","<<std::endl;
    //}
    
    //std::cout<<k.shape()[0]<<endl;
        return 0;
}
