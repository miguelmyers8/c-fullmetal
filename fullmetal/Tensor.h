//
//  Tensor.h
//  fullmetal
//
//  Created by miguel myers on 5/6/19.
//  Copyright © 2019 miguel myers. All rights reserved.
//

#ifndef Tensor_h
#define Tensor_h

#include <iostream>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

using namespace std;

template <class T>
    xt::svector<size_t> getshape(T d){
    xt::svector<size_t> y = d.shape();
    //std::copy(y.cbegin(), y.cend(), std::ostream_iterator<unsigned long>(std::cout, ", "));
    return y;
};



//template <class T>
class Tensor {
    
    public:
        xt::xarray<double> data;
        xt::svector<size_t> shape;
        xt::xarray<double> grad;
        bool requires_grad;
    
        Tensor(xt::xarray<double> a, bool b=false){
            data = a;
            requires_grad = b;
            shape = data.shape();
                }
    
        auto test();
};

//template <class T>
std::ostream& operator<<(std::ostream &strm, const Tensor &a) {
    return strm << "Tensor(" << a.data << ")";
}

//template <class T>
auto Tensor::test(){
    Tensor t1(data);
    cout<<t1.data<<endl;
    return t1;
};


#endif /* Tensor_h */
