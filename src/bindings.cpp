#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "QLearning.h"
#include "EpsilonGreedy.h"
#include "MonteCarloSearch.h"
namespace py = pybind11;

PYBIND11_MODULE(rl_env, m) {
  m.doc() = "QLearning module written in cpp";
  
  py::class_<QLearning>(m, "QLearning")
    .def(py::init<int, int, float, float>())
    .def("update", &QLearning::update, "The update function for QLearning", py::arg("state"), py::arg("action"), py::arg("reward"), py::arg("new_state"))
    .def("get_q_table", &QLearning::getQTable);
    // .def("set_q_table", &QLearning::setQTable);
  
  py::class_<EpsilonGreedy>(m, "epsilonGreedy")
    .def(py::init<float, float>(), py::arg("The beta value"), py::arg("epsilonFloor"))
    .def("choose_action", &EpsilonGreedy::chooseAction, "The function to choose an action")
    .def("get_epsilon", &EpsilonGreedy::getEpsilon, "An observer for epsilon for debugging");

  py::class_<MonteCarloSearch>(m, "MonteCarloSearch")
    .def(py::init<int, MonteCarloSearch::RewardRollout>(), py::arg("iteration limit"), py::arg("Function to randomly iterate and return reward"))
    .def("search", &MonteCarloSearch::search, "Search function", py::arg("the action space"), py::arg("QLearning object"), py::arg("current state"));
}
