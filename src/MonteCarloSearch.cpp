#include <Eigen/Dense>
#include <random>
#include <stdexcept>
#include "QLearning.h"
#include "MonteCarloSearch.h"

MonteCarloSearch::MonteCarloSearch(int iterationLimit, RewardRollout rolloutReward) 
:rng(dev()),
iterationLimit(iterationLimit),
rolloutReward(rolloutReward) {}

int MonteCarloSearch::search(const Eigen::VectorXd& actionSpace, QLearning& QLearning, int currentState) {
  if (actionSpace.size() == 0) {
    throw std::invalid_argument("Action space is empty");
  } 

  std::uniform_int_distribution<int> disui  = std::uniform_int_distribution<int>(0, actionSpace.size()-1);
  int selectedAction = disui(rng);
  std::vector<float> scores(actionSpace.size(), 0.0);
  std::vector<int> visits(actionSpace.size(), 0);
  if (actionSpace.size() > 1) {
    int iterations = 0;
    for (int currentAction = 0; currentAction < actionSpace.size(); currentAction = (currentAction + 1) % actionSpace.size()) { // Iterates over the actionspace
      if (iterations >= iterationLimit) {
        break;
      }

      float score = rolloutReward(currentState, currentAction);
      scores[currentAction] += score;
      visits[currentAction]++;
      iterations++;
    }

    float highestScore = 0.0;
    for (int i = 0; i < actionSpace.size(); i++) {
      if (visits[i] == 0) {
        scores[i] = rolloutReward(currentState, i); 
        visits[i]++;
      } // Avoiding div by 0
      float expectedScore = scores[i]/visits[i];
      if (expectedScore > highestScore) {
        selectedAction = i;
        highestScore = expectedScore;
      }
    }
  }
  return selectedAction;
}
