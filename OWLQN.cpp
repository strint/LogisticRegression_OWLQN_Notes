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

//OWLQN
//计算下降方向dir（参数的一阶梯度，虚梯度的负方向）
void OptimizerState::MakeSteepestDescDir() {

	if (l1weight == 0) {
		//l1正则化项权值为0时，查找方向dir为损失函数梯度的负方向
		scaleInto(dir, grad, -1);
	} else {
		//l1正则化项权值不为0时，根据损失函数的梯度和l1正则化项权值来确定查找方向
		for (size_t i=0; i<dim; i++) {
			if (x[i] < 0) {
				//xi<0时，|xi| = - xi，l1处的倒数为-l1weight，下降方向为梯度的反方向
				dir[i] = -grad[i] + l1weight;
			} else if (x[i] > 0) {
				//xi>0时，|xi| = xi，l1处的倒数为l1weight，下降方向为梯度的反方向
				dir[i] = -grad[i] - l1weight;
			} else {//xi == 0
				if (grad[i] < -l1weight) {
					//xi == 0，右导grad[i] + l1weight < 0，虚梯度取右导，下降方向为虚梯度的反方向，dir[i] > 0，偏向正象限
					dir[i] = -grad[i] - l1weight;
				} else if (grad[i] > l1weight) {
					//xi == 0，左导grad[i] - l1weight > 0，虚梯度取左导，下降方向为虚梯度的反方向，dir[i] < 0，偏向负象限
					dir[i] = -grad[i] + l1weight;
				} else {
					//xi == 0，左右导数都为0，下降方向为0
					dir[i] = 0;
				}
			}
		}
	}

	//记录当前的最速下降方向
	steepestDescDir = dir;
}

//lgfgs
//计算下降方向dir（参数的二阶梯度）
//lbfgs中的two loop，用过去m次的信息来近似计算Hessian矩阵的逆(进而得到当前的下降方向)
void OptimizerState::MapDirByInverseHessian() {
	int count = (int)sList.size(); //lbfgs记忆的过去的迭代结果的个数m

	if (count != 0) {
		//第一个for loop
		for (int i = count - 1; i >= 0; i--) {
			alphas[i] = -dotProduct(*sList[i], dir) / roList[i]; //不同于论文中的地方是，这里ruo的计算未取倒数，所以这里是除法；另外，这里的alpha取了负值
			addMult(dir, *yList[i], alphas[i]);
		}

		//根据lastY和lastRuo 计算了一个值，对应论文中的rj，这里保存了roList，所以使用roList[[count - 1]简化了计算
		const DblVec& lastY = *yList[count - 1];
		double yDotY = dotProduct(lastY, lastY);
		double scalar = roList[count - 1] / yDotY;
		scale(dir, scalar);

		//第二个for loop
		for (int i = 0; i < count; i++) {
			double beta = dotProduct(*yList[i], dir) / roList[i];//不同于论文中的地方是，这里ruo的计算未取倒数，所以这里是除法
			addMult(dir, *sList[i], -alphas[i] - beta);
		}
	}
}

void OptimizerState::FixDirSigns() {
	//如果存在l1正则化项
	if (l1weight > 0) {
		//dim是参数（特征）的维度数
		for (size_t i = 0; i<dim; i++) {
			//dir[i]与原来的虚梯度计算出来的方向不同的维度，置零
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

//计算的是线性查找更新步长的一部分：判断停止查找的条件中的下降方向*虚梯度[未乘以alpha]
double OptimizerState::DirDeriv() const {
	if (l1weight == 0) {
		return dotProduct(dir, grad);
	} else {
		double val = 0.0;
		for (size_t i = 0; i < dim; i++) {
			//同MakeSteepestDescDir中虚梯度的计算
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

//根据x，dir，alpha获得新的查找点newX
void OptimizerState::GetNextPoint(double alpha) {
	//获得新的查找点newX
	addMultInto(newX, x, dir, alpha);
	if (l1weight > 0) {
		for (size_t i=0; i<dim; i++) {
			//如果查找点跨了象限，置零
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

//回退的线性查找：找更新的步长（学习率）alpha
void OptimizerState::BackTrackingLineSearch() {
	//计算的是线性查找更新步长的一部分：判断停止查找的条件中的下降方向*虚梯度[未乘以alpha]
	double origDirDeriv = DirDeriv();
	// if a non-descent direction is chosen, the line search will break anyway, so throw here
	// The most likely reason for this is a bug in your function's gradient computation
	if (origDirDeriv >= 0) {
		cerr << "L-BFGS chose a non-descent direction: check your gradient!" << endl;
		exit(1);
	}

	double alpha = 1.0;
	double backoff = 0.5;

	//第一次迭代时
	if (iter == 1) {
		//alpha = 0.1;
		//backoff = 0.5;
		//计算dir的绝对值
		double normDir = sqrt(dotProduct(dir, dir));
		//将alpha、backoff设置成新的特定值
		alpha = (1 / normDir);
		backoff = 0.1;
	}

	const double c1 = 1e-4;
	double oldValue = value; //记录之前的损失值

	while (true) {
		//根据x，dir，alpha获得新的查找点newX
		GetNextPoint(alpha);
		//根据newX（即参数）来计算新的梯度newGrad、新的损失值value
		value = EvalL1();


		//计算的是线性查找更新步长的停止查找条件
		if (value <= oldValue + c1 * origDirDeriv * alpha) break;

		if (!quiet) cout << "." << flush;

		//更新alpha：如果不符合停止查找条件，步长回退，即beta^n
		alpha *= backoff;
	}

	if (!quiet) cout << endl;
}

//优化的状态迁移：更新lbfgs中两个记忆列表
void OptimizerState::Shift() {
	DblVec *nextS = NULL, *nextY = NULL;

	//lbfgs中记忆项的个数
	int listSize = (int)sList.size();

	//刚开始时，记忆项不到m，所以申请新的记忆项空间
	if (listSize < m) {
		try {
			nextS = new vector<double>(dim);
			nextY = new vector<double>(dim);
		} catch (bad_alloc) {
			m = listSize; //未分配的可用内存不够时，就使用当前能够分配的记忆项的数量作为m
			if (nextS != NULL) { //如果给S分配成功了，但Y未分配成功，就把S释放掉
				delete nextS;
				nextS = NULL;
			}
		}
	}

	//如果未分配新的S和Y，即已经有m个记忆项了
	if (nextS == NULL) {
		nextS = sList.front();
		sList.pop_front(); //弹出最老的s
		nextY = yList.front();
		yList.pop_front(); //弹出最老的y
		roList.pop_front(); //弹出最老的rou
	}

	//计算参数和梯度的差值，存入*nextS和nextY
	addMultInto(*nextS, newX, x, -1);
	addMultInto(*nextY, newGrad, grad, -1);

	//计算新的ruo，不同于论文中的地方是，这里未取倒数
	double ro = dotProduct(*nextS, *nextY); 

	//保存新的记忆项
	sList.push_back(nextS);
	yList.push_back(nextY);
	roList.push_back(ro);

	//将新的参数和梯度设为当前的参数和梯度
	x.swap(newX);
	grad.swap(newGrad);

	//迭代计数增加
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
		//减少的损失值相对于当前损失的比例
		double termCritVal = termCrit->GetValue(state, str);
		if (!quiet) {
			cout << "Iter " << setw(4) << state.iter << ":  " << setw(10) << state.value;
			cout << str.str() << flush;
		}
		//如果减少的损失值相对于当前损失的比例小于某个阈值，就停止迭代
		if (termCritVal < tol) break;

		//更新状态
		state.Shift();
	}

	if (!quiet) cout << endl;

	//将最终得到的参数存到计算结果变量中
	minimum = state.newX;
}
