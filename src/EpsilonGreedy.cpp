#include <cmath>
#include <iostream>
#include <random>
#include <Eigen/Dense>
#include "QLearning.h"
#include "EpsilonGreedy.h"

epsilonGreedy::epsilonGreedy(float betaValue) : rng(dev()){
  beta = betaValue;
    epsilon = 1.0f;

    // Initialise random
    dis = std::uniform_real_distribution<float>(0.0f, 1.0f);
  };

int epsilonGreedy::chooseAction(const Eigen::VectorXd& actionSpace, int currentState, QLearning& QLearning, int step) {
  int action;
  if (dis(rng) < epsilon) {
    // Pick randomly
    std::uniform_int_distribution<int> disui(0, actionSpace.size() - 1);
    action = disui(rng);
  } else {
    auto row = QLearning.getQTable().row(currentState);
    double maxValue = row.maxCoeff();

    std::vector<int> maxValues;
    for (int i = 0; i < row.size(); ++i) {
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

void epsilonGreedy::decay(int step) {
  epsilon = std::pow(beta, step);
};
