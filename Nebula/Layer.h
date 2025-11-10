#include "Neuron.h"
#include<random>
#pragma once
class Layer
{
	int shape_;
	std::vector<std::shared_ptr<Neuron>> neurons;
	std::shared_ptr<Layer> next_layer_;
	void forward() {
		for (int i = 0; i < next_layer_->neurons.size(); i++) {
			for (auto n:neurons) {
					next_layer_->neurons[i]->data_ = Tensor::Add(
						next_layer_->neurons[i]->data_,
						Tensor::Multiply(n->data_, n->weights_[i]));
			}
			
		}
	}
};


