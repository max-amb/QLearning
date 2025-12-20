#include <Eigen/Dense>
#include <stdexcept>
#include "QLearning.h"

QLearning::QLearning(int numberOfActions, int numberOfStates, double learningRate, double discountRate)
  : numActions(numberOfActions),
    numStates(numberOfStates),
    alpha(learningRate),
    gamma(discountRate) {
    resetQTable(); // Set QTable to all zeros
  };

void QLearning::update(int state, int action, double reward, int newState) {
  // Bounds checking
  if (state < 0 || state >= numStates || newState < 0 || newState >= numStates) {
      throw std::out_of_range("State index out of bounds");
  }
  if (action < 0 || action >= numActions) {
      throw std::out_of_range("Action index out of bounds");
  }

  // Gets the maximum value move in the row of the new state
  double maxQ = qTable.row(newState).maxCoeff();

  // Updates: Q(s, a) ← (1−α)Q(s, a) + α(R(s, a) + γmaxa′Q(s′, a′));
  qTable(state,action) = (1-alpha)*qTable(state, action) + alpha*(reward + gamma*maxQ);
};

void QLearning::resetQTable() {
  qTable = Eigen::MatrixXd::Zero(numStates, numActions);
};

/*
void QLearning::setQTable(const Eigen::MatrixXd& newQTable) {
  qTable = newQTable;
}*/

Eigen::MatrixXd& QLearning::getQTable() { return qTable; }
