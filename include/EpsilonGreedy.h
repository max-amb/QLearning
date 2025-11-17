#pragma once

#include <random>
#include <Eigen/Dense>
#include "QLearning.h"

class EpsilonGreedy {
private:
  float epsilon;
  float beta;
  std::random_device dev;
  std::mt19937 rng;
  std::uniform_real_distribution<float> dis;
  void decay(int step);

public:
  EpsilonGreedy(float betaValue);
  int chooseAction(const Eigen::VectorXd& actionSpace, int currentState, QLearning& QLearning, int step);
};

