#include "leastSquares.h"

#include <fstream>
#include <sstream>
#include <string>

using namespace std;

LeastSquaresProblem::LeastSquaresProblem(const char* matFilename, const char* bFilename) {
	ifstream matfile(matFilename);
	if (!matfile.good()) {
		cerr << "error opening matrix file " << matFilename << endl;
		exit(1);
	}

	string s;
	getline(matfile, s);
	if (!s.compare("%%MatrixMarket matrix array real general")) {
		skipEmptyAndComment(matfile, s);
		stringstream st(s);
		st >> m >> n;
		Amat.resize(m * n);

		for (size_t j=0; j<n; j++) {
			for (size_t i=0; i<m; i++) {
				float val;
				matfile >> val;
				A(i, j) = val;
			}
		}

		matfile.close();
	} else {
		matfile.close();
		cerr << "Unsupported matrix format \"" << s << "\" in " << matFilename << endl;
		exit(1);
	}

	ifstream bFile(bFilename);
	if (!bFile.good()) {
		cerr << "error opening y-value file " << bFilename << endl;
		exit(1);
	}
	getline(bFile, s);
	if (s.compare("%%MatrixMarket matrix array real general")) {
		bFile.close();
		cerr << "unsupported y-value file format \"" << s << "\" in " << bFilename << endl;
		exit(1);
	}

	skipEmptyAndComment(bFile, s);
	stringstream bst(s);
	size_t bNum, bCol;
	bst >> bNum >> bCol;
	if (bNum != m) {
		cerr << "number of y-values doesn't match number of instances in " << bFilename << endl;
		exit(1);
	} else if (bCol != 1) {
		cerr << "y-value matrix may not have more than one column" << endl;
		exit(1);
	}

	b.resize(m);
	for (size_t i=0; i<m; i++) {
		float val;
		bFile >> val;
		b[i] = val;
	}
	bFile.close();
}


double LeastSquaresObjective::Eval(const DblVec& input, DblVec& gradient) {
	static DblVec temp(problem.m);

	if (input.size() != problem.n) {
		cerr << "Error: input is not the correct size." << endl;
		exit(1);
	}

	for (size_t i=0; i<problem.m; i++) {
		temp[i] = -problem.b[i];
	}

	double value = 0.0;
	for (size_t j=0; j<problem.n; j++) {
		value += input[j] * input[j] * l2weight;
		gradient[j] = l2weight * input[j];
		for (size_t i=0; i<problem.m; i++) {
			temp[i] += input[j] * problem.A(i,j);
		}
	}

	for (size_t i=0; i<problem.m; i++) {
		if (temp[i] == 0) continue;

		value += temp[i] * temp[i];
		for (size_t j=0; j<problem.n; j++) {
			gradient[j] += problem.A(i, j) * temp[i];
		}
	}

	return 0.5 * value + 1.0;
}
