#pragma once

#include <vector>
#include <deque>
#include <iostream>

typedef std::vector<double> DblVec;

struct DifferentiableFunction {
	virtual double Eval(const DblVec& input, DblVec& gradient) = 0;
	virtual ~DifferentiableFunction() { }
};

#include "TerminationCriterion.h"

class OWLQN {
	bool quiet;
	bool responsibleForTermCrit;

public:
	TerminationCriterion *termCrit;

	OWLQN(bool quiet = false) : quiet(quiet) {
		termCrit = new RelativeMeanImprovementCriterion(5);
		responsibleForTermCrit = true;
	}

	OWLQN(TerminationCriterion *termCrit, bool quiet = false) : quiet(quiet), termCrit(termCrit) { 
		responsibleForTermCrit = false;
	}

	~OWLQN() {
		if (termCrit && responsibleForTermCrit) delete termCrit;
	}

	//寻找最小损失的过程
	//输入依次为：优化问题、初始参数、收敛时的参数（输出的结果）、l1正则化项的参数、允许的误差、limit-memory中记忆的迭代步数的数量
	void Minimize(DifferentiableFunction& function, const DblVec& initial, DblVec& minimum, double l1weight = 1.0, double tol = 1e-4, int m = 10) const;
	void SetQuiet(bool q) { quiet = q; }

};

class OptimizerState {
	friend class OWLQN;

	struct DblVecPtrDeque : public std::deque<DblVec*> {
		~DblVecPtrDeque() {
			for (size_t s = 0; s < size(); ++s) {
				if ((*this)[s] != NULL) delete (*this)[s];
			}
		}
	};

	
	DblVec x, grad, newX, newGrad, dir;//x为参数向量，grad为目标函数的梯度向量，newX为新的参数向量，dir为参数的搜索方向
	DblVec& steepestDescDir; //下降最快的下降方向 references newGrad to save memory, since we don't ever use both at the same time
	DblVecPtrDeque sList, yList; //lbfgs获得下降方向中的two-loop相关，sList记录之前m次迭代的前后两次迭代的参数的差值，yList记录之前m次迭代的的前后两次迭代的梯度的差值
	std::deque<double> roList;//lbfgs中获得下降方向中的two-loop中的rou
	std::vector<double> alphas;//lbfgs中获得下降方向中的two-loop中的alpha
	double value; //当前的目标函数的损失值
	int iter, m; //iter为优化计算的迭代次数的记录，m为limit-memory要记录的个数
	const size_t dim; //参数（特征）的维度
	DifferentiableFunction& func;//要优化的问题
	double l1weight;//l1正则化项的系数
	bool quiet; //是否输出静默

	static double dotProduct(const DblVec& a, const DblVec& b);
	static void add(DblVec& a, const DblVec& b);
	static void addMult(DblVec& a, const DblVec& b, double c);
	static void addMultInto(DblVec& a, const DblVec& b, const DblVec& c, double d);
	static void scale(DblVec& a, double b);
	static void scaleInto(DblVec& a, const DblVec& b, double c);

	void MapDirByInverseHessian();
	void UpdateDir();
	double DirDeriv() const;
	void GetNextPoint(double alpha);
	void BackTrackingLineSearch();
	void Shift();
	void MakeSteepestDescDir();
	double EvalL1();
	void FixDirSigns();
	void TestDirDeriv();

	//输入依次为：优化问题、初始参数、limit-memory中记忆的迭代步数的数量、l1正则化项的参数、是否输出静默
	OptimizerState(DifferentiableFunction& f, const DblVec& init, int m, double l1weight, bool quiet) 
		: x(init), grad(init.size()), newX(init), newGrad(init.size()), dir(init.size()), steepestDescDir(newGrad), alphas(m), iter(1), m(m), dim(init.size()), func(f), l1weight(l1weight), quiet(quiet) {
		// 初始化：x初始化为初始参数向量，grad初始化为长度和参数向量长度一样的空向量，
		        //newX初始化为初始参数向量，newGrad初始化为长度和参数向量长度一样的空向量，
		        //dir初始化为长度和参数向量长度一样的空向量，steepestDescDir初始化为和newGrad一样的空向量，
			    //alphas初始化为长度为m的空向量，dim初始化为init的长度，
			    //func初始化为优化问题f，l1正则化项系数初始化为l1weight
			    //quiet初始化为quiet
			if (m <= 0) {
				std::cerr << "m must be an integer greater than zero." << std::endl;
				exit(1);
			}
			//更新梯度、计算损失
			value = EvalL1();
			grad = newGrad;
	}

public:
	const DblVec& GetX() const { return newX; }
	const DblVec& GetLastX() const { return x; }
	const DblVec& GetGrad() const { return newGrad; }
	const DblVec& GetLastGrad() const { return grad; }
	const DblVec& GetLastDir() const { return dir; }
	double GetValue() const { return value; }
	int GetIter() const { return iter; }
	size_t GetDim() const { return dim; }
};
