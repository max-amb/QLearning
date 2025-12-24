#pragma once

#include <Eigen/Dense>
#include <random>
#include "QLearning.h"

// For our rolloutReward function
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

class MonteCarloSearch {
public: // Needs to be on top for RewardRollout
  using RewardRollout = std::function<float(int, int)>;
  MonteCarloSearch(int iterationLimit, RewardRollout rolloutReward);
  int search(const Eigen::VectorXd& actionSpace, QLearning& QLearning, int currentState);

private:
  int iterationLimit; // Replacement for time limit
  std::random_device dev;
  std::mt19937 rng;
  RewardRollout rolloutReward;
};
