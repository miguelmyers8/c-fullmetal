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

auto aa = xt::random::randn<double> ({2, 2});
auto bb = xt::random::randn<double> ({2, 2});

xt::xarray<double> k = {{ 11, 12, 13 }};
xt::xarray<double> l = {  1,  2,  3 };
auto v = k+l;

int main(int argc, const char * argv[]) {
    Tensor t1(aa);
    //cout<<t1<<endl;
    Tensor x = add(t1,t1);
    cout<<x<<endl;
        return 0;
}
