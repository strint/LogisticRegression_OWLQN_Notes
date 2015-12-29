#include "OWLQN.h"

#include "TerminationCriterion.h"

#include <vector>
#include <deque>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace std;

double OptimizerState::dotProduct(const DblVec& a, const DblVec& b) {
	double result = 0;
	for (size_t i=0; i<a.size(); i++) {
		result += a[i] * b[i];
	}
	return result;
}

void OptimizerState::addMult(DblVec& a, const DblVec& b, double c) {
	for (size_t i=0; i<a.size(); i++) {
		a[i] += b[i] * c;
	}
}

void OptimizerState::add(DblVec& a, const DblVec& b) {
	for (size_t i=0; i<a.size(); i++) {
		a[i] += b[i];
	}
}

void OptimizerState::addMultInto(DblVec& a, const DblVec& b, const DblVec& c, double d) {
	for (size_t i=0; i<a.size(); i++) {
		a[i] = b[i] + c[i] * d;
	}
}

void OptimizerState::scale(DblVec& a, double b) {
	for (size_t i=0; i<a.size(); i++) {
		a[i] *= b;
	}
}

void OptimizerState::scaleInto(DblVec& a, const DblVec& b, double c) {
	for (size_t i=0; i<a.size(); i++) {
		a[i] = b[i] * c;
	}
}

//计算下降方向dir（参数的一阶梯度）
void OptimizerState::MakeSteepestDescDir() {

	if (l1weight == 0) {
		//l1正则化项权值为0时，查找方向dir为损失函数梯度的负方向
		scaleInto(dir, grad, -1);
	} else {
		//l1正则化项权值不为0时，根据损失函数的梯度和l1正则化项权值来确定查找方向
		for (size_t i=0; i<dim; i++) {
			if (x[i] < 0) {
				//xi<0时，右导一定小于0，左导中xi的符号为负，虚梯度取右导，下降方向为虚梯度的反方向
				dir[i] = -grad[i] + l1weight;
			} else if (x[i] > 0) {
				//xi>0时，左导一定大于0，左导中xi的符号为正，虚梯度取左导，下降方向为虚梯度的反方向
				dir[i] = -grad[i] - l1weight;
			} else {//xi == 0
				if (grad[i] < -l1weight) {
					//xi == 0，右导<0，虚梯度取右导，下降方向为虚梯度的反方向
					dir[i] = -grad[i] - l1weight;
				} else if (grad[i] > l1weight) {
					//xi == 0，左导>0，虚梯度取左导，下降方向为虚梯度的反方向
					dir[i] = -grad[i] + l1weight;
				} else {
					//xi == 0，左右岛数都为0，下降方向为0
					dir[i] = 0;
				}
			}
		}
	}

	//当前的最速下降方向
	steepestDescDir = dir;
}

//计算下降方向dir（参数的二阶梯度）
//lbfgs中的two loop，用过去m次的信息来近似计算Hessian矩阵的逆(进而得到当前的下降方向)
void OptimizerState::MapDirByInverseHessian() {
	int count = (int)sList.size();

	if (count != 0) {
		for (int i = count - 1; i >= 0; i--) {
			alphas[i] = -dotProduct(*sList[i], dir) / roList[i];
			addMult(dir, *yList[i], alphas[i]);
		}

		const DblVec& lastY = *yList[count - 1];
		double yDotY = dotProduct(lastY, lastY);
		double scalar = roList[count - 1] / yDotY;
		scale(dir, scalar);

		for (int i = 0; i < count; i++) {
			double beta = dotProduct(*yList[i], dir) / roList[i];
			addMult(dir, *sList[i], -alphas[i] - beta);
		}
	}
}

void OptimizerState::FixDirSigns() {
	if (l1weight > 0) {
		for (size_t i = 0; i<dim; i++) {
			if (dir[i] * steepestDescDir[i] <= 0) {
				dir[i] = 0;
			}
		}
	}
}

void OptimizerState::UpdateDir() {
	MakeSteepestDescDir();
	MapDirByInverseHessian();
	FixDirSigns();

#ifdef _DEBUG
	TestDirDeriv();
#endif
}

void OptimizerState::TestDirDeriv() {
	double dirNorm = sqrt(dotProduct(dir, dir));
	double eps = 1.05e-8 / dirNorm;
	GetNextPoint(eps);
	double val2 = EvalL1();
	double numDeriv = (val2 - value) / eps;
	double deriv = DirDeriv();
	if (!quiet) cout << "  Grad check: " << numDeriv << " vs. " << deriv << "  ";
}

double OptimizerState::DirDeriv() const {
	if (l1weight == 0) {
		return dotProduct(dir, grad);
	} else {
		double val = 0.0;
		for (size_t i = 0; i < dim; i++) {
			if (dir[i] != 0) { 
				if (x[i] < 0) {
					val += dir[i] * (grad[i] - l1weight);
				} else if (x[i] > 0) {
					val += dir[i] * (grad[i] + l1weight);
				} else if (dir[i] < 0) {
					val += dir[i] * (grad[i] - l1weight);
				} else if (dir[i] > 0) {
					val += dir[i] * (grad[i] + l1weight);
				}
			}
		}

		return val;
	}
}

void OptimizerState::GetNextPoint(double alpha) {
	addMultInto(newX, x, dir, alpha);
	if (l1weight > 0) {
		for (size_t i=0; i<dim; i++) {
			if (x[i] * newX[i] < 0.0) {
				newX[i] = 0.0;
			}
		}
	}
}

double OptimizerState::EvalL1() {
	//根据新的X（即参数）来计算新的梯度newGrad、新的损失值loss
	double val = func.Eval(newX, newGrad);
	//如果l1正则化项的参数为正，损失加上l1正则化项的部分
	if (l1weight > 0) {
		for (size_t i=0; i<dim; i++) {
			val += fabs(newX[i]) * l1weight;
		}
	}

	//返回损失值
	return val;
}

void OptimizerState::BackTrackingLineSearch() {
	double origDirDeriv = DirDeriv();
	// if a non-descent direction is chosen, the line search will break anyway, so throw here
	// The most likely reason for this is a bug in your function's gradient computation
	if (origDirDeriv >= 0) {
		cerr << "L-BFGS chose a non-descent direction: check your gradient!" << endl;
		exit(1);
	}

	double alpha = 1.0;
	double backoff = 0.5;
	if (iter == 1) {
		//alpha = 0.1;
		//backoff = 0.5;
		double normDir = sqrt(dotProduct(dir, dir));
		alpha = (1 / normDir);
		backoff = 0.1;
	}

	const double c1 = 1e-4;
	double oldValue = value;

	while (true) {
		GetNextPoint(alpha);
		//更新梯度、计算损失
		value = EvalL1();

		if (value <= oldValue + c1 * origDirDeriv * alpha) break;

		if (!quiet) cout << "." << flush;

		alpha *= backoff;
	}

	if (!quiet) cout << endl;
}

void OptimizerState::Shift() {
	DblVec *nextS = NULL, *nextY = NULL;

	int listSize = (int)sList.size();

	if (listSize < m) {
		try {
			nextS = new vector<double>(dim);
			nextY = new vector<double>(dim);
		} catch (bad_alloc) {
			m = listSize;
			if (nextS != NULL) {
				delete nextS;
				nextS = NULL;
			}
		}
	}

	if (nextS == NULL) {
		nextS = sList.front();
		sList.pop_front();
		nextY = yList.front();
		yList.pop_front();
		roList.pop_front();
	}

	addMultInto(*nextS, newX, x, -1);
	addMultInto(*nextY, newGrad, grad, -1);
	double ro = dotProduct(*nextS, *nextY);

	sList.push_back(nextS);
	yList.push_back(nextY);
	roList.push_back(ro);

	x.swap(newX);
	grad.swap(newGrad);

	iter++;
}

//寻找最小损失的过程
//输入依次为：优化问题、初始参数、收敛时的参数（输出的结果）、l1正则化项的参数、允许的误差、limit-memory中记忆的迭代步数的数量
void OWLQN::Minimize(DifferentiableFunction& function, const DblVec& initial, DblVec& minimum, double l1weight, double tol, int m) const {
	//输入依次为：优化问题、初始参数、limit-memory中记忆的迭代步数的数量、l1正则化项的参数、是否输出静默
	OptimizerState state(function, initial, m, l1weight, quiet);

	if (!quiet) {
		cout << setprecision(4) << scientific << right;
		cout << endl << "Optimizing function of " << state.dim << " variables with OWL-QN parameters:" << endl;
		cout << "   l1 regularization weight: " << l1weight << "." << endl;
		cout << "   L-BFGS memory parameter (m): " << m << endl;
		cout << "   Convergence tolerance: " << tol << endl;
		cout << endl;
		cout << "Iter    n:  new_value    (conv_crit)   line_search" << endl << flush;
		cout << "Iter    0:  " << setw(10) << state.value << "  (***********) " << flush;
	}

	ostringstream str;
	termCrit->GetValue(state, str);

	while (true) {
		//更新search direction
		state.UpdateDir();
		//查找step size
		state.BackTrackingLineSearch();

		//判断是否满足终止条件
		ostringstream str;
		double termCritVal = termCrit->GetValue(state, str);
		if (!quiet) {
			cout << "Iter " << setw(4) << state.iter << ":  " << setw(10) << state.value;
			cout << str.str() << flush;
		}

		if (termCritVal < tol) break;

		//更新状态
		state.Shift();
	}

	if (!quiet) cout << endl;

	minimum = state.newX;
}
