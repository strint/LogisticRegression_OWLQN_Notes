#pragma once

#include <fstream>
#include <string>

#include "OWLQN.h"

struct LeastSquaresObjective;

class LeastSquaresProblem {
	std::vector<float> Amat;
	std::vector<float> b;
	size_t m, n;
	
	void skipEmptyAndComment(std::ifstream& file, std::string& s) {
		do {
			std::getline(file, s);
		} while (s.size() == 0 || s[0] == '%');
	}

	friend struct LeastSquaresObjective;

public:
	LeastSquaresProblem(size_t m, size_t n) : Amat(m * n), b(m), m(m), n(n) { }

	LeastSquaresProblem(const char* matfile, const char* bFile);

	float A(size_t i, size_t j) const {
		return Amat[i + m * j];
	}

	float& A(size_t i, size_t j) {
		return Amat[i + m * j];
	}

	size_t NumFeats() const { return n; }
	size_t NumInstances() const { return m; }
};

struct LeastSquaresObjective : public DifferentiableFunction {
	const LeastSquaresProblem& problem;
	const double l2weight;

	LeastSquaresObjective(const LeastSquaresProblem& p, double l2weight = 0) : problem(p), l2weight(l2weight) { }

	double Eval(const DblVec& input, DblVec& gradient);
};
