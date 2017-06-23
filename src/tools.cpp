#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  std::size_t size = estimations.size();
  VectorXd rmse = VectorXd::Zero(4);
  if (!size || size != ground_truth.size()) return rmse;
  for (int i = 0; i < size; ++i) {
    VectorXd diff = estimations[i] - ground_truth[i];
    rmse += diff.cwiseProduct(diff);
  }
  rmse /= size;
  return rmse.cwiseSqrt();
}

double Tools::CalculateProportionAbove(const vector<double> &nis, double threshold) {
  std::size_t size = nis.size();
  if (!size) return 0;
  int count = 0;
  for (int i = 0; i < size; ++i)
    if (nis[i] >= threshold) ++count;
  return (double) count / size;
}