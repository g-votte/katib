# Document about how to add a new algorithm in Katib

## Implement a new algorithm and use it in Katib

### Implement the algorithm

The design of Katib follows the `ask-and-tell` pattern:

> They often follow a pattern a bit like this: 1. ask for a new set of parameters 1. walk to the Experiment and program in the new parameters 1. observe the outcome of running the Experiment 1. walk back to your laptop and tell the optimizer about the outcome 1. go to step 1

When an Experiment is created, one algorithm service will be created. Then Katib asks for new sets of parameters via `GetSuggestions` GRPC call. After that, Katib creates new trials according to the sets and observe the outcome. When the trials are finished, Katib tells the metrics of the finished trials to the algorithm, and ask another new sets.

The new algorithm needs to implement `Suggestion` service defined in [api.proto](../pkg/apis/manager/v1beta1/api.proto). One sample algorithm looks like:

```python
from pkg.apis.manager.v1beta1.python import api_pb2
from pkg.apis.manager.v1beta1.python import api_pb2_grpc
from pkg.suggestion.v1beta1.internal.search_space import HyperParameter, HyperParameterSearchSpace
from pkg.suggestion.v1beta1.internal.trial import Trial, Assignment
from pkg.suggestion.v1beta1.hyperopt.base_service import BaseHyperoptService
from pkg.suggestion.v1beta1.base_health_service import HealthServicer


# Inherit SuggestionServicer and implement GetSuggestions.
class HyperoptService(
        api_pb2_grpc.SuggestionServicer, HealthServicer):
    def ValidateAlgorithmSettings(self, request, context):
        # Optional, it is used to validate algorithm settings defined by users.
        pass
    def GetSuggestions(self, request, context):
        # Convert the Experiment in GRPC request to the search space.
        # search_space example:
        #   HyperParameterSearchSpace(
        #       goal: MAXIMIZE,
        #       params: [HyperParameter(name: param-1, type: INTEGER, min: 1, max: 5, step: 0),
        #                HyperParameter(name: param-2, type: CATEGORICAL, list: cat1, cat2, cat3),
        #                HyperParameter(name: param-3, type: DISCRETE, list: 3, 2, 6),
        #                HyperParameter(name: param-4, type: DOUBLE, min: 1, max: 5, step: )]
        #   )
        search_space = HyperParameterSearchSpace.convert(request.experiment)
        # Convert the trials in GRPC request to the trials in algorithm side.
        # trials example:
        #   [Trial(
        #       assignment: [Assignment(name=param-1, value=2),
        #                    Assignment(name=param-2, value=cat1),
        #                    Assignment(name=param-3, value=2),
        #                    Assignment(name=param-4, value=3.44)],
        #       target_metric: Metric(name="metric-2" value="5643"),
        #       additional_metrics: [Metric(name=metric-1, value=435),
        #                            Metric(name=metric-3, value=5643)],
        #   Trial(
        #       assignment: [Assignment(name=param-1, value=3),
        #                    Assignment(name=param-2, value=cat2),
        #                    Assignment(name=param-3, value=6),
        #                    Assignment(name=param-4, value=4.44)],
        #       target_metric: Metric(name="metric-2" value="3242"),
        #       additional_metrics: [Metric(name=metric=1, value=123),
        #                            Metric(name=metric-3, value=543)],
        trials = Trial.convert(request.trials)
        #--------------------------------------------------------------
        # Your code here
        # Implement the logic to generate new assignments for the given request number.
        # For example, if request.request_number is 2, you should return:
        # [
        #   [Assignment(name=param-1, value=3),
        #    Assignment(name=param-2, value=cat2),
        #    Assignment(name=param-3, value=3),
        #    Assignment(name=param-4, value=3.22)
        #   ],
        #   [Assignment(name=param-1, value=4),
        #    Assignment(name=param-2, value=cat4),
        #    Assignment(name=param-3, value=2),
        #    Assignment(name=param-4, value=4.32)
        #   ],
        # ]
        list_of_assignments = your_logic(search_space, trials, request.request_number)
        #--------------------------------------------------------------
        # Convert list_of_assignments to
        return api_pb2.GetSuggestionsReply(
            trials=Assignment.generate(list_of_assignments)
        )
```

### Make a GRPC server for the algorithm

Create a package under [cmd/suggestion](../cmd/suggestion). Then create the main function and Dockerfile. The new GRPC server should serve in port 6789.

Here is an example: [cmd/suggestion/hyperopt](../cmd/suggestion/hyperopt).
Then build the Docker image.

### Use the algorithm in Katib.

Update the [Katib config](../manifests/v1beta1/components/controller/katib-config.yaml)
and [Katib config patch](../manifests/v1beta1/installs/katib-standalone/katib-config-patch.yaml)
with the new algorithm entity:

```diff
  suggestion: |-
    {
      "tpe": {
        "image": "docker.io/kubeflowkatib/suggestion-hyperopt"
      },
      "random": {
        "image": "docker.io/kubeflowkatib/suggestion-hyperopt"
      },
+     "<new-algorithm-name>": {
+       "image": "image built in the previous stage"
+     }
    }
```

Learn more about Katib config in the
[Kubeflow documentation](https://www.kubeflow.org/docs/components/katib/katib-config/)

### Contribute the algorithm to Katib

If you want to contribute the algorithm to Katib, you could add unit test and/or
e2e test for it in the CI and submit a PR.

#### Unit Test

Here is an example [test_hyperopt_service.py](../test/suggestion/v1beta1/test_hyperopt_service.py):

```python
import grpc
import grpc_testing
import unittest

from pkg.apis.manager.v1beta1.python import api_pb2_grpc
from pkg.apis.manager.v1beta1.python import api_pb2

from pkg.suggestion.v1beta1.hyperopt.service import HyperoptService

class TestHyperopt(unittest.TestCase):
    def setUp(self):
        servicers = {
            api_pb2.DESCRIPTOR.services_by_name['Suggestion']: HyperoptService()
        }

        self.test_server = grpc_testing.server_from_dictionary(
            servicers, grpc_testing.strict_real_time())


if __name__ == '__main__':
    unittest.main()
```

You can setup the GRPC server using `grpc_testing`, then define your own test cases.

#### E2E Test (Optional)

E2e tests help Katib verify that the algorithm works well.
Follow below steps to add your algorithm (Suggestion) to the Katib CI
(replace `<name>` with your Suggestion name):

1. Submit a PR to add a new ECR private registry to the AWS
   [`ECR_Private_Registry_List`](https://github.com/kubeflow/testing/blob/master/aws/IaC/CDK/test-infra/config/static_config/ECR_Resources.py#L18).
   Registry name should follow the pattern: `katib/v1beta1/suggestion-<name>`

1. Create a new Experiment YAML in the [examples/v1beta1](../examples/v1beta1)
   with the new algorithm.

1. Update [`setup-katib.sh`](../test/scripts/v1beta1/setup-katib.sh)
   script to modify `katib-config.yaml` with the new test Suggestion image name.
   For example:

   ```sh
   sed -i -e "s@docker.io/kubeflowkatib/suggestion-<name>@${ECR_REGISTRY}/${REPO_NAME}/v1beta1/suggestion-<name>@" ${CONFIG_PATCH}
   ```

1. Add a new two steps in the CI workflow
   ([test/workflows/components/workflows-v1beta1.libsonnet](../test/workflows/components/workflows-v1beta1.libsonnet))
   to build and run the new Suggestion:

```diff
. . .
                  {
                    name: "build-suggestion-hyperopt",
                    template: "build-suggestion-hyperopt",
                  },
                  {
                    name: "build-suggestion-chocolate",
                    template: "build-suggestion-chocolate",
                  },
+                 {
+                   name: "build-suggestion-<name>",
+                   template: "build-suggestion-<name>",
+                 },
. . .
                  {
                    name: "run-tpe-e2e-tests",
                    template: "run-tpe-e2e-tests",
                  },
                  {
                    name: "run-grid-e2e-tests",
                    template: "run-grid-e2e-tests",
                  },
+                 {
+                   name: "run-<name>-e2e-tests",
+                   template: "run-<name>-e2e-tests",
+                 },
. . .
            $.parts(namespace, name, overrides).e2e(prow_env, bucket).buildTemplate("build-suggestion-hyperopt", kanikoExecutorImage, [
              "/kaniko/executor",
              "--dockerfile=" + katibDir + "/cmd/suggestion/hyperopt/v1beta1/Dockerfile",
              "--context=dir://" + katibDir,
              "--destination=" + registry + "/katib/v1beta1/suggestion-hyperopt:$(PULL_BASE_SHA)",
            ]),  // build suggestion hyperopt
            $.parts(namespace, name, overrides).e2e(prow_env, bucket).buildTemplate("build-suggestion-chocolate", kanikoExecutorImage, [
              "/kaniko/executor",
              "--dockerfile=" + katibDir + "/cmd/suggestion/chocolate/v1beta1/Dockerfile",
              "--context=dir://" + katibDir,
              "--destination=" + registry + "/katib/v1beta1/suggestion-chocolate:$(PULL_BASE_SHA)",
            ]),  // build suggestion chocolate
+           $.parts(namespace, name, overrides).e2e(prow_env, bucket).buildTemplate("build-suggestion-<name>", kanikoExecutorImage, [
+             "/kaniko/executor",
+             "--dockerfile=" + katibDir + "/cmd/suggestion/<name>/v1beta1/Dockerfile",
+             "--context=dir://" + katibDir,
+             "--destination=" + registry + "/katib/v1beta1/suggestion-<name>:$(PULL_BASE_SHA)",
+           ]),  // build suggestion <name>
. . .
            $.parts(namespace, name, overrides).e2e(prow_env, bucket).buildTemplate("run-tpe-e2e-tests", testWorkerImage, [
              "test/scripts/v1beta1/run-e2e-experiment.sh",
              "examples/v1beta1/tpe-example.yaml",
            ]),  // run TPE algorithm
            $.parts(namespace, name, overrides).e2e(prow_env, bucket).buildTemplate("run-grid-e2e-tests", testWorkerImage, [
              "test/scripts/v1beta1/run-e2e-experiment.sh",
              "examples/v1beta1/grid-example.yaml",
            ]),  // run grid algorithm
+           $.parts(namespace, name, overrides).e2e(prow_env, bucket).buildTemplate("run-<name>-e2e-tests", testWorkerImage, [
+             "test/scripts/v1beta1/run-e2e-experiment.sh",
+             "examples/v1beta1/<name>-example.yaml",
+           ]),  // run <name> algorithm
. . .
```
