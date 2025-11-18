#include <Eigen/Dense>
#include <stdexcept>
#include "QLearning.h"

QLearning::QLearning(int numberOfActions, int numberOfStates, double learningRate, double discountRate)
  : numActions(numberOfActions),
    numStates(numberOfStates),
    alpha(learningRate),
    gamma(discountRate) {
    resetQTable();
  };

void QLearning::update(int state, int action, double reward, int newState) {
    if (state < 0 || state >= numStates || newState < 0 || newState >= numStates) {
        throw std::out_of_range("State index out of bounds");
    }
    if (action < 0 || action >= numActions) {
        throw std::out_of_range("Action index out of bounds");
    }
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
