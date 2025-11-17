#include <Eigen/Dense>

class SWF {
  private:
    int state;
    static const int num_actions = 2;
    static const int num_states = 7;
    static constexpr double discount = 0.9;
    int states[num_states] = {0, 1, 2, 3, 4, 5, 6};
    int actions[num_actions] = {-1, 1};
    Eigen::MatrixXd transition_probabilities[num_actions];
    Eigen::MatrixXd rewards;

  public:
    SWF() : state(3) {
      // Initialise transition matrices
      for (int a = 0; a < num_actions; a++) {
        transition_probabilities[a] = Eigen::MatrixXd::Zero(num_states, num_states);
      }

      for (int a = 0; a < num_actions; ++a) {
          for (int s = 0; s < num_states; ++s) {
              int intended = std::clamp(s + actions[a], 0, num_states - 1);
              int opposite = std::clamp(s - actions[a], 0, num_states - 1);
              transition_probabilities[a](s, intended) += 0.5;
              transition_probabilities[a](s, opposite) += 0.17;
              transition_probabilities[a](s, s) += 0.33;
          }
      }

      // Initialise rewards
      rewards = Eigen::MatrixXd::Zero(num_states, num_states);
      rewards.row(num_states - 1).setOnes();

    }; 
};
