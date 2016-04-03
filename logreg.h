#pragma once

#include <deque>
#include <vector>
#include <cmath>
#include <iostream>

#include "OWLQN.h"

//主要做的是把数据(MatrixMarket格式)读进来、存起来
class LogisticRegressionProblem {
	std::deque<size_t> indices; //样本i的维度值，indeces[instance_starts[i]]到indices[instance_starts[i-1] - 1]
	std::deque<float> values;//样本i的与indices中维度值对应的特征值，values[instance_starts[i]]到values[instance_starts[i-1] - 1]
	std::deque<size_t> instance_starts;//样本i在indices和values中的起始位置，instance_starts[i]
	std::deque<bool> labels;//样本i的label，labels[i]，label为bool
	size_t numFeats;//样本的维数

public:
	LogisticRegressionProblem(size_t numFeats) : numFeats(numFeats) {
		instance_starts.push_back(0);
	}

	LogisticRegressionProblem(const char* mat, const char* labels);
	void AddInstance(const std::deque<size_t>& inds, const std::deque<float>& vals, bool label);
	void AddInstance(const std::vector<float>& vals, bool label);
	double ScoreOf(size_t i, const std::vector<double>& weights) const;

	bool LabelOf(size_t i) const {
		return labels[i];
	}

	//输入是样本i的i、1-分类正确的概率、梯度向量，用来使用损失函数的非正则化项部分来更新梯度
	void AddMultTo(size_t i, double mult, std::vector<double>& vec) const {
		if (labels[i]) mult *= -1; //乘以负的标签值(-label[i])
		//对于样本i的各个维度index，用该维度对于的特征值*mult去更新梯度向量的维度index
		for (size_t j=instance_starts[i]; j < instance_starts[i+1]; j++) {
			size_t index = (indices.size() > 0) ? indices[j] : j - instance_starts[i];
			vec[index] += mult * values[j];//乘以特征值values
		}
	}

	//样本个数
	size_t NumInstances() const {
		return labels.size();
	}

	//特征向量的维度
	size_t NumFeats() const {
		return numFeats;
	}
};

struct LogisticRegressionObjective : public DifferentiableFunction {
	//存储了样本数据
	const LogisticRegressionProblem& problem;
	const double l2weight;

	LogisticRegressionObjective(const LogisticRegressionProblem& p, double l2weight = 0) : problem(p), l2weight(l2weight) { }

	//用来计算当前的参数input情况下的损失和梯度向量
	//体现了优化问题的损失函数和损失函数的梯度
	double Eval(const DblVec& input, DblVec& gradient);

};
