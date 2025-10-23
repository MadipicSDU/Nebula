#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <iostream>
#include<iomanip>
#pragma once
class Tensor : public std::enable_shared_from_this<Tensor>
{
	float data_;
	float grad_ = 0;
	std::vector<std::shared_ptr<Tensor>> prev_;
	std::function<void()> backprop_;
	
public:
	Tensor(float data) : data_(data) {};
	Tensor(float data, std::vector<std::shared_ptr<Tensor>> children, std::function<void()> func) 
		: data_(data), prev_(children), backprop_(func) {};
	float GetData() const { return data_; }
	float GetGrad() const { return grad_; }
private:
	template<typename T>
	static std::shared_ptr<Tensor> EnsureTensor(const T& value) {
		if constexpr (std::is_same_v<T, std::shared_ptr<Tensor>>) {
			return value;
		}
		else if constexpr (std::is_arithmetic_v<T>) {
			return std::make_shared<Tensor>(static_cast<float>(value));
		}
		else {
			return nullptr;
		}
	}
public:

	template<typename T1, typename T2>
	static std::shared_ptr<Tensor> Add(const T1& a, const T2& b) {
		auto ta = EnsureTensor(a);
		auto tb = EnsureTensor(b);
		float result = ta->GetData() + tb->GetData();
		auto out = std::make_shared<Tensor>(result);
		out->prev_ = { ta, tb };
		std::weak_ptr<Tensor> wout = out;
		out->backprop_ = [ta, tb, wout]() {
			auto out_ptr = wout.lock();
			if (out_ptr) {
				ta->grad_ += 1.0f * out_ptr->grad_;
				tb->grad_ += 1.0f * out_ptr->grad_;
			}
			};
		return out;
	}
	template<typename T1, typename T2>
	static std::shared_ptr<Tensor> Multiply(const T1& a, const T2& b) {
		auto ta = EnsureTensor(a);
		auto tb = EnsureTensor(b);
		float result = ta->GetData() * tb->GetData();
		auto out = std::make_shared<Tensor>(result);
		out->prev_ = { ta, tb };
		std::weak_ptr<Tensor> wout = out;
		out->backprop_ = [ta, tb, wout]() {
			auto out_ptr = wout.lock();
			if (out_ptr) {
				ta->grad_ += tb->GetData() * out_ptr->grad_;
				tb->grad_ += ta->GetData() * out_ptr->grad_;
			}
			};
		return out;
	}
	template<typename T1, typename T2>
	static std::shared_ptr<Tensor> Subtract(const T1& a, const T2& b) {
		auto ta = EnsureTensor(a);
		auto tb = EnsureTensor(b);
		float result = ta->GetData() - tb->GetData();
		auto out = std::make_shared<Tensor>(result);
		out->prev_ = { ta, tb };
		std::weak_ptr<Tensor> wout = out;
		out->backprop_ = [ta, tb, wout]() {
			auto out_ptr = wout.lock();
			if (out_ptr) {
				ta->grad_ += 1.0f * out_ptr->grad_;
				tb->grad_ += -1.0f * out_ptr->grad_;
			}
			};
		return out;
	}
	template<typename T1, typename T2>
	static std::shared_ptr<Tensor> Divide(const T1& a, const T2& b) {
		auto ta = EnsureTensor(a);
		auto tb = EnsureTensor(b);
		float result = ta->GetData() / tb->GetData();
		auto out = std::make_shared<Tensor>(result);
		out->prev_ = { ta, tb };
		std::weak_ptr<Tensor> wout = out;
		out->backprop_ = [ta, tb, wout]() {
			auto out_ptr = wout.lock();
			if (out_ptr) {
				ta->grad_ += (1.0f / tb->GetData()) * out_ptr->grad_;
				tb->grad_ += (-ta->GetData() / (tb->GetData() * tb->GetData())) * out_ptr->grad_;
			}
			};
		return out;
	}
	template<typename T1, typename T2>
	static std::shared_ptr<Tensor> Power(const T1& a, const T2& b) {
		auto ta = EnsureTensor(a);
		auto tb = EnsureTensor(b);
		float result = std::pow(ta->GetData(), tb->GetData());
		auto out = std::make_shared<Tensor>(result);
		out->prev_ = { ta, tb };
		std::weak_ptr<Tensor> wout = out;
		out->backprop_ = [ta, tb, wout]() {
			auto out_ptr = wout.lock();
			if (out_ptr) {
				ta->grad_ += (tb->GetData() * std::pow(ta->GetData(), tb->GetData() - 1)) * out_ptr->grad_;
				tb->grad_ += (std::log(ta->GetData()) * std::pow(ta->GetData(), tb->GetData())) * out_ptr->grad_;
			}
			};
		return out;
	}
	template<typename T>
	static std::shared_ptr<Tensor> Sigmoid(const T& a) {
		auto ta = EnsureTensor(a);
		float result = 1 / (1 + std::exp(-(ta->data_)));
		std::cout << std::exp(-(ta->data_));
		auto out = std::make_shared<Tensor>(result);
		out->prev_ = { ta };
		std::weak_ptr<Tensor> wout = out;
		out->backprop_ = [ta, wout, result]() {
			auto out_ptr = wout.lock();
			if (out_ptr) {
				
				ta->grad_ += (result) * (1 - result) * out_ptr->grad_;
			}
			};
		return out;
	}
	template<typename T>
	static std::shared_ptr<Tensor> Relu(const T& a) {
		auto ta = EnsureTensor(a);
		float result = (ta->data_ > 0) ? ta->data_ : 0;
		auto out = std::make_shared<Tensor>(result);
		out->prev_ = { ta };
		std::weak_ptr<Tensor> wout = out;
		out->backprop_ = [ta, wout, result]() {
			auto out_ptr = wout.lock();
			if (out_ptr) {
				ta->grad_ += (result > 0) ? 1 : 0 * out_ptr->grad_;
			}
			};
		return out;
	}
	void Backward() {
		grad_ = 1.0f;
		std::vector<std::shared_ptr<Tensor>> topo;
		std::vector<std::shared_ptr<Tensor>> visited;

		std::function<void(std::shared_ptr<Tensor>)> build_topo;
		build_topo = [&](std::shared_ptr<Tensor> v) {
			if (std::find(visited.begin(), visited.end(), v) == visited.end()) {
				visited.push_back(v);
				for (auto& child : v->prev_) {
					build_topo(child);
				}
				topo.push_back(v);
			}
			};

		build_topo(shared_from_this());
		std::reverse(topo.begin(), topo.end());

		for (auto& node : topo) {
			if (node->backprop_) node->backprop_();
		}
	}

	friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
		os << "Tensor(data=" << t.data_ << ", grad=" << t.grad_ << ")";
		return os;
	}
};


