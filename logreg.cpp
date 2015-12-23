#include "logreg.h"
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

void skipEmptyAndComment(ifstream& file, string& s) {
	do {
		getline(file, s);
	} while (s.size() == 0 || s[0] == '%');
}

//导入样本数据
LogisticRegressionProblem::LogisticRegressionProblem(const char* matFilename, const char* labelFilename) {
	ifstream matfile(matFilename);
	if (!matfile.good()) {
		cerr << "error opening matrix file " << matFilename << endl;
		exit(1);
	}
	string s;
	getline(matfile, s);
	//MatrixMarket是一个文件格式(http://math.nist.gov/MatrixMarket/formats.html)，这是这种格式文件的第一行
	if (!s.compare("%%MatrixMarket matrix coordinate real general")) {
		skipEmptyAndComment(matfile, s);

		stringstream st(s);
		size_t numIns, numNonZero;
		st >> numIns >> numFeats >> numNonZero; //行数，列数，非0数据个数

		vector<deque<size_t> > rowInds(numIns);
		vector<deque<float> > rowVals(numIns);
		for (size_t i = 0; i < numNonZero; i++) {
			size_t row, col;
			float val;
			matfile >> row >> col >> val; //行号，列号，数值
			row--;
			col--;
			rowInds[row].push_back(col);
			rowVals[row].push_back(val);
		}

		matfile.close();

		ifstream labfile(labelFilename);
		getline(labfile, s);
		if (s.compare("%%MatrixMarket matrix array real general")) {
			cerr << "unsupported label file format in " << labelFilename << endl;
			exit(1);
		}

		skipEmptyAndComment(labfile, s);
		stringstream labst(s);
		size_t labNum, labCol;
		labst >> labNum >> labCol;
		if (labNum != numIns) {
			cerr << "number of labels doesn't match number of instances in " << labelFilename << endl;
			exit(1);
		} else if (labCol != 1) {
			cerr << "label matrix may not have more than one column" << endl;
			exit(1);
		}

		instance_starts.push_back(0);

		for (size_t i=0; i<numIns; i++) {
			int label;
			labfile >> label;
			bool bLabel;
			switch (label) {
					case 1:
						bLabel = true;
						break;

					case -1:
						bLabel = false;
						break;

					default:
						cerr << "illegal label: must be 1 or -1" << endl;
						exit(1);
			}

			AddInstance(rowInds[i], rowVals[i], bLabel);//样本i的特征向量和label
		}

		labfile.close();
	} else if (!s.compare("%%MatrixMarket matrix array real general")) {
		skipEmptyAndComment(matfile, s);
		stringstream st(s);
		size_t numIns;
		st >> numIns >> numFeats;

		vector<vector<float> > rowVals(numIns);

		for (size_t j=0; j<numFeats; j++) {
			for (size_t i=0; i<numIns; i++) {
				float val;
				matfile >> val;
				rowVals[i].push_back(val);
			}
		}

		matfile.close();

		ifstream labfile(labelFilename);
		getline(labfile, s);
		if (s.compare("%%MatrixMarket matrix array real general")) { 
			cerr << "unsupported label file format in " << labelFilename << endl;
			exit(1);
		}

		skipEmptyAndComment(labfile, s);
		stringstream labst(s);
		size_t labNum, labCol;
		labst >> labNum >> labCol;
		if (labNum != numIns) {
			cerr << "number of labels doesn't match number of instances in " << labelFilename << endl;
			exit(1);
		} else if (labCol != 1) {
			cerr << "label matrix may not have more than one column" << endl;
			exit(1);
		}

		instance_starts.push_back(0);
		for (size_t i=0; i<numIns; i++) {
			int label;
			labfile >> label;
			bool bLabel;
			switch (label) {
					case 1:
						bLabel = true;
						break;

					case -1:
						bLabel = false;
						break;

					default:
						cerr << "illegal label: must be 1 or -1" << endl;
						exit(1);
			}

			AddInstance(rowVals[i], bLabel);
		}

		labfile.close();
	} else {
		cerr << "unsupported matrix file format in " << matFilename << endl;
		exit(1);
	}
}

//添加一个样本的数据
void LogisticRegressionProblem::AddInstance(const deque<size_t>& inds, const deque<float>& vals, bool label) {
	for (size_t i=0; i<inds.size(); i++) {
		indices.push_back(inds[i]);
		values.push_back(vals[i]);
	}//当前样本的inds.size()个特征维的下标及其对应的特征值
	instance_starts.push_back(indices.size());//下一个样本数据的开始位置（在indices和values中的下标）
	labels.push_back(label);//当前样本的label
}

void LogisticRegressionProblem::AddInstance(const vector<float>& vals, bool label) {
	for (size_t i=0; i<vals.size(); i++) {
		values.push_back(vals[i]);
	}
	instance_starts.push_back(values.size()); 
	labels.push_back(label);
}

//计算yi*（W*X + b)
double LogisticRegressionProblem::ScoreOf(size_t i, const vector<double>& weights) const {
	double score = 0;
	for (size_t j=instance_starts[i]; j < instance_starts[i+1]; j++) {
		double value = values[j];
		size_t index = (indices.size() > 0) ? indices[j] : j - instance_starts[i];
		score += weights[index] * value;
	}//计算样本i的各个维度的特征值与权重的乘积的和，叫score
	if (!labels[i]) score *= -1; //如果样本i的label是-1，则将score取反，乘以了样本i的label
	return score;
}

//用来计算当前的参数情况下的损失和梯度向量
//input是参数向量
double LogisticRegressionObjective::Eval(const DblVec& input, DblVec& gradient) {
	double loss = 1.0; //为什么要初始化为1？

	//用来使用损失函数的正则化项部分来更新梯度
	//正则化项部分的loss和gradient
	for (size_t i=0; i<input.size(); i++) {
		loss += 0.5 * input[i] * input[i] * l2weight;//0.5 * C * wi^2的和，累加损失
		gradient[i] = l2weight * input[i];//C * wi，当前的梯度(正则化项部分)
	}

	for (size_t i =0 ; i<problem.NumInstances(); i++) {
		double score = problem.ScoreOf(i, input);

		//insProb是样本i被正确分类到yi的概率
		double insLoss, insProb;
		if (score < -30) {
			insLoss = -score;//损失取-score，当-score比较大时，log(1.0 + exp(-score))约等于-score
			insProb = 0;//当前模型分类正确的概率为0
		} else if (score > 30) {//score大于30时
			insLoss = 0;//损失为0，当score比较大时，log(1.0 + exp(-score))约等于0
			insProb = 1;//当前模型分类正确的概率为1
		} else {//当score在-30到30之间时
			double temp = 1.0 + exp(-score);
			insLoss = log(temp);
			insProb = 1.0/temp;
		}
		loss += insLoss;//累加损失

		//用来使用损失函数的非正则化项部分来更新梯度
		//输入是样本i的i、1-分类正确的概率、梯度向量
		problem.AddMultTo(i, 1.0 - insProb, gradient);
	}

	return loss;
}
