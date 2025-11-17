#include <Eigen/Dense>
#include "QLearning.h"

QLearning::QLearning(int numberOfActions, int numberOfStates, double learningRate, double discountRate)
  : numActions(numberOfActions),
    numStates(numberOfStates),
    alpha(learningRate),
    gamma(discountRate) {
    resetQTable();
  };

void QLearning::update(int state, int action, double reward, int newState) {
    double maxQ = qTable.row(newState).maxCoeff();
    qTable(state,action) += alpha*(reward + gamma*maxQ - qTable(state, action));
 };

void QLearning::resetQTable() {
    qTable = Eigen::MatrixXd::Zero(numStates, numActions);
};

/*
void QLearning::setQTable(const Eigen::MatrixXd& newQTable) {
  qTable = newQTable;
}*/

Eigen::MatrixXd& QLearning::getQTable() { return qTable; }
