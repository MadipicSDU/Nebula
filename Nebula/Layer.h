#include "Tensor.h"
#pragma once
class Layer
{
	int shape;
	std::vector<std::shared_ptr<Tensor>> neurouns;
	std::function<void()> activation_func;
	
};


