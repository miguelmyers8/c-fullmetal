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
#include <optional>

auto aa = xt::random::randn<double> ({2, 2});
auto bb = xt::random::randn<double> ({2, 2});

xt::xarray<double> k = {{ 11, 12, 13 }};
xt::xarray<double> l = {{  1,  1,  1 }};
xt::xarray<double> ll = {{6}};
xt::xarray<double> f = {{ 11, 12, 16 }};
auto v = k+l;


int main(int argc, const char * argv[]) {
    
    
    Tensor t2(k,true);
    Tensor t3(f,true);
    Tensor x = t3+t2;
    
    for (const Dependancies& i : x.depend_on){ // access by const reference
        std::cout << i << ' '<<endl;
        cout << i.grad_fn(l)<< endl;
        
    }

        return 0;
}
