import hyperopt.hp
import hyperopt.pyll.stochastic
import kubernetes as k8s
import bayesian
import json
import time

class ConfigMap:
    def __init__(self):
        k8s.config.load_incluster_config()

    def patch(self, configmap_name, configmap_namespace, data):
        # https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/CoreV1Api.md#patch_namespaced_config_map
        body = {
            "kind": "ConfigMap",
            "apiVersion": "v1",
            "metadata": { "labels": { "date": str(int(time.time())) } },
            "data": data
        }

        api_instance = k8s.client.CoreV1Api(k8s.client.ApiClient())
        name = configmap_name
        namespace = configmap_namespace
        pretty = 'true'

        return api_instance.patch_namespaced_config_map(name, namespace, body, pretty=pretty)

    def patch_jvm(self, configmap_name, configmap_namespace, data):
        # https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/CoreV1Api.md#patch_namespaced_config_map

        params = []
        if '-Xmx' in data:
            params.append('-Xmx'+str(data['-Xmx'])+'m')

        if '-Xnojit' in data and data['-Xnojit']:
            params.append('-Xnojit')
            
        if '-Xnoaot' in data and data['-Xnoaot']:
            params.append('-Xnoaot')

        body = {
            "kind": "ConfigMap",
            "apiVersion": "v1",
            "metadata": {"labels": {"date": str(int(time.time()))}},
            "data": {'jvm.options': '\n'.join(params)}
        }

        api_instance = k8s.client.CoreV1Api(k8s.client.ApiClient())
        name = configmap_name
        namespace = configmap_namespace
        pretty = 'true'

        return api_instance.patch_namespaced_config_map(name, namespace, body, pretty=pretty)

    def patch_deployment(self, deployment_name, deployment_namespace, data):
        api_instance = k8s.client.AppsV1Api(k8s.client.ApiClient())

        name = deployment_name
        namespace = deployment_namespace
        pretty = 'true'

        body = {
            "kind": "Deployment",
            "apiVersion": "v1",
            "metadata": {"labels": {"date": str(int(time.time()))}},
            "spec": {"template": {"spec": {"containers": [
                {
                    "name": "proxy",
                    "resources": {
                        "limits": {
                            "cpu": "0.1",
                            "memory": "256Mi"
                        }
                    }
                },
                {
                    "name": "service",
                    "resources": {
                        "limits": {
                            "cpu": "0.1",
                            "memory": "256Mi"
                        }
                    }
                }
            ]}}}
        }
        api_instance.list_namespaced_deployment('default')
        api_instance.patch_namespaced_deployment(name, namespace, body)

        return api_instance.patch_namespaced_config_map(name, namespace, body, pretty=pretty)



def load_search_space(search_space_path):
    search_space = SearchSpace({})

    print('\nloading search space')
    with open(search_space_path) as json_file:
        data = json.load(json_file)
        for item in data:
            print('\t', item)
            search_space.add_to_domain(
                key=item.get('key', None),
                lower=item.get('lower', None),
                upper=item.get('upper', None),
                options=item.get('options', None),
                step=item.get('step', None),
                type=item.get('type', None)
            )

    return search_space

class SearchSpace:
    def __init__(self, domain=None):
        if not domain:
            domain = {}
        self.domain = domain

    def get_lower(self, key):
        return self.domain[key][0]

    def get_upper(self, key):
        return self.domain[key][1]

    def get_options(self, key):
        return self.domain[key][2]

    def get_step(self, key):
        return self.domain[key][3]

    def get_type(self, key):
        return self.domain[key][4]

    def add_to_domain(self, key, type=None, lower=None, step=None, upper=None, options=None):
        """
        type: str, int, float, bool
        """
        t = self.convert_type_from_str(type)

        try:
            step = float(step)
            if step < 0:
                step  = abs(step)
            elif step == 0:
                step = None
        except (TypeError, ValueError):
            step = None


        if self.check_arguments(t, lower, upper, options):
            self.domain.update({key: (lower, upper, options, step, type)})

    def convert_type_from_str(self, type):
        if isinstance(type, str):
            if 'INT' == type.upper() or 'INTEGER' == type.upper():
                return int
            if 'FLOAT' == type.upper() or 'DOUBLE' == type.upper:
                return float
            if 'STR' == type.upper() or 'STRING' == type.upper():
                return str
            if 'BOOL' == type.upper() or 'BOOLEAN' == type.upper():
                return bool
            raise TypeError(f'type={type} not known')
        return type

    def check_arguments(self, type, lower, upper, options):

        if type == int:
            if not (isinstance(lower, int) and isinstance(upper, int)):
                raise TypeError(f"both lower and upper params should be int for type={int}")
            if options != None:
                raise TypeError(f"type={int} doesn't use options")
            return True
        elif type == float:
            if not (isinstance(lower, float) and isinstance(upper, float)):
                raise TypeError(f"both lower and upper params should be float for type={type}")
            if options != None:
                raise TypeError(f"type={float} doesn't use options")
            return True
        elif type == bool:
            if lower != None or upper != None or options != None:
                raise TypeError(f"type={type} doesn't need any other argument")
            return True
        elif type == str:
            if lower != None and upper != None:
                raise TypeError(f"type={type} only needs arg options")
            return True
        else:
            raise TypeError(f"cannot handle type={type}")

        return True

    def dimension(self, key):
        if self.get_type(key) == int.__name__ or self.get_type(key) == float.__name__:
            if self.get_step(key):
                dimension = hyperopt.hp.quniform(key, self.get_lower(key), self.get_upper(key), self.get_step(key))
            else:
                dimension = hyperopt.hp.uniform(key, self.get_lower(key), self.get_upper(key))
        elif self.get_type(key) == bool.__name__:
            dimension = hyperopt.hp.choice(key, [True, False])
        elif self.get_type(key) == str.__name__:
            dimension = hyperopt.hp.choice(key, self.get_options(key))
        else:
            raise TypeError(f'cannot handle type={type}')

        return {key: dimension}

    def search_space(self):
        space = {}
        for key, value in self.domain.items():
            space.update(self.dimension(key))
        return space

    def sampling(self):

        #
        # sample = hyperopt.pyll.stochastic.sample(self.search_space())
        #
        if not bayesian.space:
            bayesian.init(self.search_space())

        sample = bayesian.get()

        return self.sample_values_to_str(sample)

    def update_model(self, metric):
        bayesian.put(metric)

    def cast_value(self, key, value):
        # 0: lower bound
        # 1: upper bound
        # 2: options
        # 3: type
        return self.convert_type_from_str(self.get_type(key))(value)

    def sample_values_to_str(self, sample):
        str_sample = {}

        for key, value in sample.items():
            value = self.cast_value(key, value)
            str_sample[key] = str(value)


        return str_sample
