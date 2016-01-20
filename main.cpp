#include <iostream>
#include <deque>
#include <fstream>

#include "OWLQN.h"
#include "leastSquares.h"
#include "logreg.h"

using namespace std;

void printUsageAndExit() {
	cout << "Orthant-Wise Limited-memory Quasi-Newton trainer" << endl;
	cout << "trains L1-regularized logistic regression or least-squares models" << endl << endl;
	cout << "usage: feature_file label_file regWeight output_file [options]" << endl;
	cout << "  feature_file   input feature matrix in Matrix Market format (mxn real coordinate or array)" << endl;
	cout << "                   rows represent features for each instance" << endl;
	cout << "  label_file     input instance labels in Matrix Market format (mx1 real array)" << endl;
	cout << "                   rows contain single real value" << endl;
	cout << "                   for logistic regression problems, value must be 1 or -1" << endl;
	cout << "  regWeight      coefficient of l1 regularizer" << endl;
	cout << "  output_file    output weight vector in Matrix Market format (1xm real array)" << endl << endl;
	cout << "options:" << endl;
	cout << "  -ls            use least squares formulation (logistic regression is default)" << endl;
	cout << "  -q             quiet.  Suppress all output" << endl;
	cout << "  -tol <value>   sets convergence tolerance (default is 1e-4)" << endl;
	cout << "  -m <value>     sets L-BFGS memory parameter (default is 10)" << endl;
	cout << "  -l2weight <value>" << endl;
	cout << "                 sets L2 regularization weight (default is 0)" << endl;
	cout << endl;
	system("pause");
	exit(0);
}

void printVector(const DblVec &vec, const char* filename) {
	ofstream outfile(filename);
	if (!outfile.good()) {
		cerr << "error opening matrix file " << filename << endl;
		exit(1);
	}
	outfile << "%%MatrixMarket matrix array real general" << endl;
	outfile << "1 " << vec.size() << endl;
	for (size_t i=0; i<vec.size(); i++) {
		outfile << vec[i] << endl;
	}
	outfile.close();
}

int main(int argc, char* argv[]) {

	//输入测参数至少包括程序本身的名字、feature_file、label_file、regWeight（coefficient of l1 regularizer）、output_file五个参数
	//如果输入的参数少于5个或者第一个参数中包含help字符，则打印帮助并退出
	if (argc < 5 || !strcmp(argv[1], "-help") || !strcmp(argv[1], "--help") ||
		!strcmp(argv[1], "-h") || !strcmp(argv[1], "-usage")) {
			printUsageAndExit();
	}

	//读入feature_file、label_file、regWeight（coefficient of l1 regularizer）、output_file
	const char* feature_file = argv[1];
	const char* label_file = argv[2];
	double regweight = atof(argv[3]);//l1正则化项
	const char* output_file = argv[4];

	if (regweight < 0) {
		cout << "L1 regularization weight must be non-negative." << endl;
		exit(1);
	}

	//给出默认值
	bool leastSquares = false, quiet = false;
	double tol = 1e-4, l2weight = 0;
	int m = 10;

	//对于可选的配置信息
	for (int i=5; i<argc; i++) {
		if (!strcmp(argv[i], "-ls")) leastSquares = true; //判断是否使用least square
		else if (!strcmp(argv[i], "-q")) quiet = true; //判断是否静默输出
		else if (!strcmp(argv[i], "-tol")) {
			//读取tolerance
			++i;
			if (i >= argc || (tol = atof(argv[i])) <= 0) {
				cout << "-tol (convergence tolerance) flag requires 1 positive real argument." << endl;
				exit(1);
			}
		} else if (!strcmp(argv[i], "-l2weight")) {
			//读取l2正则化项的权重
			++i;
			if (i >= argc || (l2weight = atof(argv[i])) < 0) {
				cout << "-l2weight flag requires 1 non-negative real argument." << endl;
				exit(1);
			}
		}	else if (!strcmp(argv[i], "-m")) {
			//读取记忆项的个数
			++i;
			if (i >= argc || (m = atoi(argv[i])) == 0) {
				cout << "-m (L-BFGS memory param) flag requires 1 positive int argument." << endl;
				exit(1);
			}
		} else {
			cerr << "unrecognized argument: " << argv[i] << endl;
			exit(1);
		}
	}

	if (!quiet) {
		cout << argv[0] << " called with arguments " << endl << "   ";
		for (int i=1; i<argc; i++) {
			cout << argv[i] << " ";
		}
		cout << endl;
	}

	DifferentiableFunction *obj;
	size_t size;
	if (leastSquares) {
		LeastSquaresProblem *prob = new LeastSquaresProblem(feature_file, label_file);
		obj = new LeastSquaresObjective(*prob, l2weight);
		size = prob->NumFeats(); 
	} else {
		//将数据导入到逻辑回归问题中
		LogisticRegressionProblem *prob = new LogisticRegressionProblem(feature_file, label_file);
		obj = new LogisticRegressionObjective(*prob, l2weight);
		size = prob->NumFeats(); 
	}

	//size为特征的维度，init为初始参数值向量，ans为结果参数值向量
	DblVec init(size), ans(size);

	OWLQN opt(quiet);
	//输入依次是LogisticRegressionObjective（包含了样本数据、l2正则化项的系数、损失函数）、
	//参数的初始化值、参数最终的结果、l1正则化项的系数、允许的误差、lbfgs的记忆的项数
	opt.Minimize(*obj, init, ans, regweight, tol, m);

	int nonZero = 0;
	for (size_t i = 0; i<ans.size(); i++) {
		if (ans[i] != 0) nonZero++;
	}

	if (!quiet) cout << "Finished with optimization.  " << nonZero << "/" << size << " non-zero weights." << endl;

	printVector(ans, output_file);

	return 0;
}
