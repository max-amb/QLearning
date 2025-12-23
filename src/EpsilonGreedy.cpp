#include <cmath>
#include <random>
#include <stdexcept>
#include <Eigen/Dense>
#include "QLearning.h"
#include "EpsilonGreedy.h"
#include "MonteCarloSearch.h"

EpsilonGreedy::EpsilonGreedy(float beta, float epsilonFLoor)
  : rng(dev()),
  beta(beta),
  epsilonFloor(epsilonFLoor) {
  epsilon = 0.5f;

  // Initialise random
  dis = std::uniform_real_distribution<float>(0.0f, 1.0f);
};

// Probably shouldn't be here but oh well
int EpsilonGreedy::chooseAction(const Eigen::VectorXd& actionSpace, int currentState, QLearning& QLearning, MonteCarloSearch& MCS, int step) {
  // Bounds checking
  if (actionSpace.size() == 0) {
    throw std::invalid_argument("Action space is empty");
  }
  if (currentState < 0 || currentState >= QLearning.getQTable().rows()) {
    throw std::out_of_range("Current state index out of bounds");
  }
  
  int action;
  if (dis(rng) < epsilon) {
    action = MCS.search(actionSpace, QLearning, currentState);
  } else {
    auto row = QLearning.getQTable().row(currentState);
    double maxValue = row.maxCoeff();

    std::vector<int> maxValues;
    for (int i = 0; i < row.size() && i < actionSpace.size(); ++i) {
        if (row[i] == maxValue) {
            maxValues.push_back(i);
        }
    }
    std::uniform_int_distribution<int> disui(0, maxValues.size() - 1);
    action = maxValues[disui(rng)];
  }
  decay(step);
  return action;
};

void EpsilonGreedy::decay(int step) {
  if (epsilon == epsilonFloor) { return; }
  float decayed = std::pow(beta, step);
  epsilon = decayed < epsilonFloor ? epsilonFloor : decayed;
};
