#include "Tensor.h"

int main() {
    auto a = std::make_shared<Tensor>(2.0f);
    auto b = std::make_shared<Tensor>(3.0f);
	auto c = Tensor::Add(a, b); 
	auto d = Tensor::Multiply(c, 4.0f);
	auto e = Tensor::Subtract(d, 5.0f);
    e = Tensor::Power(e, d);
	e = Tensor::Divide(e, 2.0f);
	auto f = Tensor::Sigmoid(e);
	auto g = Tensor::Relu(f);
	g->Backward();
    std::cout << "a: " << *a << std::endl;
    std::cout << "b: " << *b << std::endl;
    std::cout << "c: " << *c << std::endl;
    std::cout << "d: " << *d << std::endl;
	std::cout << "e: " << *e << std::endl;
	std::cout << "f: " << *f << std::endl;
	std::cout << "g: " << *g << std::endl;
    return 0;
}