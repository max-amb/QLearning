#pragma once

#include <random>
#include <Eigen/Dense>
#include "QLearning.h"
#include "MonteCarloSearch.h"

class EpsilonGreedy {
private:
  float epsilon;
  float beta;
  float epsilonFloor;
  std::random_device dev;
  std::mt19937 rng;
  std::uniform_real_distribution<float> dis;
  void decay(int step);

public:
  EpsilonGreedy(float betaValue, float epsilonFloor);
  int chooseAction(const Eigen::VectorXd& actionSpace, int currentState, QLearning& QLearning, MonteCarloSearch& MCS, int step);
  float getEpsilon();
};

