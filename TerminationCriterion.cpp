#include "TerminationCriterion.h"

#include "OWLQN.h"

#include <limits>
#include <iomanip>
#include <cmath>

using namespace std;

//相对平均提高标准
double RelativeMeanImprovementCriterion::GetValue(const OptimizerState& state, std::ostream& message) {
	double retVal = numeric_limits<double>::infinity();

	//如果已经记录了5个以上损失值了
	if (prevVals.size() > 5) {
		//取队首的值
		double prevVal = prevVals.front();
		//如果已经有10个值了，就把队首的值删掉
		if (prevVals.size() == 10) prevVals.pop_front();
		//用队首的值减去最新的值，除以队列的长度，得到队列的平均提高
		double averageImprovement = (prevVal - state.GetValue()) / prevVals.size();
		//用队列的平均提高除以当前值得到提高相对于当前值的比例
		double relAvgImpr = averageImprovement / fabs(state.GetValue());
		message << setprecision(4) << scientific << right;
		message << "  (" << setw(10) << relAvgImpr << ") " << flush;
		//将提高比例保存到retVal中
		retVal = relAvgImpr;
	} else {
		message << "  (wait for five iters) " << flush;
	}

	//将当前迭代的损失添加到损失列表尾部
	prevVals.push_back(state.GetValue());
	//返回提高的比例
	return retVal;
}
