# Copyright 2021 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import optuna

from pkg.apis.manager.v1beta1.python import api_pb2
from pkg.apis.manager.v1beta1.python import api_pb2_grpc
from pkg.suggestion.v1beta1.internal.constant import INTEGER, DOUBLE, CATEGORICAL, DISCRETE, MAX_GOAL
from pkg.suggestion.v1beta1.internal.search_space import HyperParameterSearchSpace
from pkg.suggestion.v1beta1.internal.trial import Trial, Assignment
from pkg.suggestion.v1beta1.internal.base_health_service import HealthServicer


logger = logging.getLogger(__name__)


class OptunaService(api_pb2_grpc.SuggestionServicer, HealthServicer):

    def __init__(self):
        super(OptunaService, self).__init__()
        self.study = None
        self.search_space = None
        self.recorded_trial_names = []
        self.trial_katib_name_to_optuna_number = {}

    def GetSuggestions(self, request, context):
        """
        Main function to provide suggestion.
        """
        if self.study is None:
            self.search_space = HyperParameterSearchSpace.convert(request.experiment)
            self.study = self._create_study(request.experiment.spec.algorithm, self.search_space)

        trials = Trial.convert(request.trials)

        if len(trials) != 0:
            self._tell(trials)

        list_of_assignments = self._ask(request.request_number)
        # new_trials = self.base_service.getSuggestions(trials, request.request_number)

        return api_pb2.GetSuggestionsReply(
            parameter_assignments=Assignment.generate(list_of_assignments)
        )

    @staticmethod
    def _create_study(algorithm_spec, search_space):
        algorithm_name = algorithm_spec.algorithm_name
        if algorithm_name == "tpe":
            sampler = optuna.samplers.TPESampler()
        elif algorithm_name == "multivariate-tpe":
            sampler = optuna.samplers.TPESampler(multivariate=True)
        else:
            raise ValueError("Unknown algorithm name: {}".format(algorithm_name))

        direction = "maximize" if search_space.goal == MAX_GOAL else "minimize"
        study = optuna.create_study(sampler=sampler, direction=direction)

        return study

    def _ask(self, request_number):
        list_of_assignments = []
        for _ in range(request_number):
            optuna_trial = self.study.ask()
            assignments = []

            for param in self.search_space.params:
                if param.type == INTEGER:
                    value = optuna_trial.suggest_int(param.name, int(param.min), int(param.max))
                elif param.type == DOUBLE:
                    value = optuna_trial.suggest_float(param.name, float(param.min), float(param.max))
                elif param.type == CATEGORICAL or param.type == DISCRETE:
                    value = optuna_trial.suggest_categorical(param.name, param.list)

                assignment = Assignment(param.name, value)
                assignments.append(assignment)

            assignments_key = self._get_assignments_key(assignments)
            self.trial_katib_name_to_optuna_number[assignments_key] = optuna_trial.number

            list_of_assignments.append(assignments)

        return list_of_assignments

    def _tell(self, trials):
        for trial in trials:
            if trial.name not in self.recorded_trial_names:
                self.recorded_trial_names.append(trial.name)

                assignments_key = self._get_assignments_key(trial.assignments)
                trial_number = self.trial_katib_name_to_optuna_number[assignments_key]
                del self.trial_katib_name_to_optuna_number[assignments_key]

                value = trial.target_metric.value
                self.study.tell(trial_number, value)

    @staticmethod
    def _get_assignments_key(assignments):
        assignments = sorted(assignments, key=lambda a: a.name)
        assignments_str = [str(a) for a in assignments]
        return ",".join(assignments_str)
