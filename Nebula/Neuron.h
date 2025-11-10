#include"Tensor.h"
#pragma once
class Neuron
{
public:
	std::shared_ptr<Tensor> data_;
	std::vector<std::shared_ptr<Tensor>> weights_;
	std::shared_ptr<Tensor> bias_;
	int input_size_ = 0;
	std::function<void()> activation_function_;
	void activation() {
		data_ = Tensor::Sigmoid(Tensor::Add(data_,bias_));
	}
	void initialize_weights(int input_size) {
		input_size_ = input_size;
		std::default_random_engine generator;
		std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
		for (int i = 0; i < input_size; i++) {
			weights_.push_back(std::make_shared<Tensor>(distribution(generator)));
		}
		bias_ = std::make_shared<Tensor>(distribution(generator));
	}
};

