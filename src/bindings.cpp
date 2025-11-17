#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "QLearning.h"
#include "EpsilonGreedy.h"
namespace py = pybind11;

PYBIND11_MODULE(rl_env, m) {
    m.doc() = "QLearning module written in cpp";
    
    py::class_<QLearning>(m, "QLearning")
      .def(py::init<int, int, float, float>())
      .def("update", &QLearning::update, "The update function for QLearning", py::arg("state"), py::arg("action"), py::arg("reward"), py::arg("new_state"))
      .def("get_q_table", &QLearning::getQTable);
      // .def("set_q_table", &QLearning::setQTable);
    
    py::class_<epsilonGreedy>(m, "epsilonGreedy")
      .def(py::init<float>())
      .def("choose_action", &epsilonGreedy::chooseAction, "The function to choose an action");
}
