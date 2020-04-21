import hyperopt.hp
import hyperopt.pyll.stochastic
import sys
import os
import kubernetes as k8s
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
            "data": data
        }

        api_instance = k8s.client.CoreV1Api(k8s.client.ApiClient())
        name = configmap_name
        namespace = configmap_namespace
        pretty = 'true'

        return api_instance.patch_namespaced_config_map(name, namespace, body, pretty=pretty)

class Deployment:
    # update deployment
    # DEPLOYMENT_NAME=tuning-deployment
    # curl -sSk \
    #   -X PATCH \
    #   -d @- \
    #   -H "Authorization: Bearer $KUBE_TOKEN" \
    #   -H 'Accept: application/json' \
    #   -H 'Content-Type: application/strategic-merge-patch+json' \
    #   https://$KUBERNETES_SERVICE_HOST:$KUBERNETES_PORT_443_TCP_PORT/apis/apps/v1/namespaces/$NAMESPACE/deployments/$DEPLOYMENT_NAME <<'EOF'
    # {
    #   "kind": "Deployment",
    #   "apiVersion": "v1",
    #   "metadata": {
    #     "labels": {
    # 		"config": "2"
    # 	}
    #   }
    # }
    # EOF
    pass

class SearchSpace:
    def __init__(self, domain=None):
        if not domain:
            domain = {}
        self.domain = domain
        self.history = {}

    def add_to_domain(self, key, type=None, lower=None, upper=None, options=None):
        """
        type: str, int, float, bool
        """
        t = self.convert_type_from_str(type)
        if self.check_arguments(t, lower, upper, options):
            self.domain.update({key: (lower, upper, options, type)})

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
        lower = self.domain[key][0]
        upper = self.domain[key][1]
        options = self.domain[key][2]
        t = self.domain[key][3]
        if t == int.__name__:
            dimension = hyperopt.hp.quniform(key, lower, upper, 1)
        elif t == float.__name__:
            dimension = hyperopt.hp.uniform(key, lower, upper)
        elif t == bool.__name__:
            dimension = hyperopt.hp.choice(key, [True, False])
        elif t == str.__name__:
            dimension = hyperopt.hp.choice(key, options)
        else:
            raise TypeError(f'cannot handle type={type}')

        return {key: dimension}

    def search_space(self):
        space = {}
        for key, value in self.domain.items():
            space.update(self.dimension(key))
        return space

    def sampling(self, label):

        if not label in self.history:
            self.history[label] = set()

        sample = hyperopt.pyll.stochastic.sample(self.search_space())

        self.history[label].add(json.dumps(sample))

        return sample, len(self.history[label])

    def sample_values_to_str(self, sample):
        str_sample = {}

        for key, value in sample.items():
            if isinstance(value, int):
                str_sample[key] = str(value)
            elif isinstance(value, float):
                str_sample[key] = str(value)
            elif isinstance(value, bool):
                str_sample[key] = str(value).upper()
            else:
                str_sample[key] = value

        return str_sample

def main():
    config_path = os.environ['CONFIGMAP_PATH']
    searchspace_path = os.environ['CONFIGMAP_SEARCHSPACE']
    wait_time = int(os.environ['WAIT_TIME'])
    print(wait_time)

    while True:
        envvar = {}
        files = [f for f in os.listdir(config_path) if os.path.isfile(os.path.join(config_path, f))]

        for f in files:
            envvar[f] = os.environ.get(f,'')

        search_space = SearchSpace({})
        config_map = ConfigMap()


        with open(searchspace_path) as json_file:
            data = json.load(json_file)
            for item in data:
                search_space.add_to_domain(
                    key=item.get('key', None),
                    lower=item.get('lower', None),
                    upper=item.get('upper', None),
                    options=item.get('options', None),
                    type=item.get('type', None)
                )

        sample, length = search_space.sampling('test')
        sample = search_space.sample_values_to_str(sample)
        print('new config >>> ', sample)

        response = config_map.patch(os.environ['CONFIGMAP_NAME'], os.environ['NAMESPACE'], sample)

        time.sleep(wait_time)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
