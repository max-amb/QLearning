#pragma once

#include <Eigen/Dense>

class QLearning {
  private:
  int numActions;
  int numStates;
  double alpha;
  double gamma;
  Eigen::MatrixXd qTable;
public:
  QLearning(int numberOfActions, int numberOfStates, double learningRate, double discountRate);
  void update(int state, int action, double reward, int newState);
  void resetQTable();
  // use Eigen::Ref<MatrixType> so we do not need setQTable
  // void setQTable(const Eigen::MatrixXd& newQTable);
  Eigen::MatrixXd& getQTable();
};
