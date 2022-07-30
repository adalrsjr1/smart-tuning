import json
import re
import sys

apps = {
'acmeair' : 'resources/trace-acmeair-2021-09-14T19 46 28.json',
'quarkus' : 'resources/trace-quarkus-2021-09-14T19 46 43.json',
'daytrader' : 'resources/trace-daytrader-2021-09-15T23 26 02.json',
'daytrader_fw' : 'resources/trace-daytrader-2021-09-22T02 42 28.json'
}

def iteration(line: str) -> dict:
    line = re.sub(r'{"_id":{"\$oid":"[a-z0-9]+"},', '{', line)
    line = re.sub(r'\{\"\$numberLong\":\"(?P<N>[0-9]+)\"}', '\g<N>', line)
    return json.loads(line)

def filter_out_workloads(workload: str, iterations: list[dict]) -> list[iteration]:
    filtered_iterations: list[dict] = []
    for iteration in iterations:
        if iteration['curr_workload']['name'] == iteration['ctx_workload']['name'] and \
           iteration['curr_workload']['name'] == workload:
               filtered_iterations.append(iteration)

    return filtered_iterations

def format_output(app_name: str, workload_name: str, json_data0: dict, json_dataN: dict):
    title = f'{app_name}:{workload_name}'
    header = f'{"app level":30} {"parameter":15} {"inital cfg":>30} {"final cfg":>30}'
    magic_span = 15
    print(title)
    print('='*(len(header)+magic_span))
    print(header)
    print('='*(len(header)+magic_span))
    app_levels = ['container runtime', 'app server', 'jvm']
    for idx, level in enumerate(json_data0.keys()):
        #print('-'*len(level))
        #print(app_levels[idx])
        #print('-'*len(level))
        for attr0, attrN in zip(list(json_data0[level].items()), list(json_dataN[level].items())):
            attr_name = attr0[0]
            attr0_value = attr0[1]
            attrN_value = attrN[1]
            print(f'{app_levels[idx]:20} |{attr_name:35} |{attr0_value:>30} |{attrN_value:>30}')
    print('='*(len(header)+magic_span))
    print()

def format_output_latex(app_name: str, workload_name: str, json_data0: dict, json_dataN: dict):
    title = re.sub(r'_', '\\_', f'{app_name.title()}:{workload_name.title()}')
    print("""
    \\begin{{table*}}[h]
        \\centering
        \\begin{{tabular}}{{llcc}}
        \\multicolumn{{4}}{{l}}{{\\textbf{{{0}}}}} \\\\ \\hline
    """.format(title))

    header = '{0:30} & {1:15} & {2:>30} & {3:>30} \\\\ \hline'.format(
            '\emph{App level}',
            '\emph{Parameter}',
            '\emph{Initical Cfg}',
            '\emph{Final Cfg}')
    magic_span = 15
    print(header)
    app_levels = ['container runtime', 'app server', 'jvm']
    for idx, level in enumerate(json_data0.keys()):
        for attr0, attrN in zip(list(json_data0[level].items()), list(json_dataN[level].items())):
            attr_name = re.sub(r'_', '\\_', attr0[0])
            attr0_value = re.sub(r'_', '\\_', str(attr0[1]))
            attrN_value = re.sub(r'_', '\\_', str(attrN[1]))
            print(f'{app_levels[idx]:20} & {attr_name:35} &{attr0_value:>30} &{attrN_value:>30} \\\\')
    print('\hline')
    print("""
        \\end{{tabular}}
        \\caption{{}}
        \\label{{tab:{0}-{1}}}
    \\end{{table*}}
    """.format(app_name, re.sub(r'_', '-', workload_name)))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: extract_configs app_name workload')
        print('app_names:\n\tacmeair\n\tdaytrader\n\tdatrader_fw\n\tquarkus\n')
        print('workload:\n\tacmeair and quarkus: workload_50, workload_100, workload_200 \
                \n\tdaytrader: workload_5, workload_10, workload_50\n\tdaytrader_fw: workload_jsp, workload_jsf')
        exit(1)
    app_name = sys.argv[1]
    workload = sys.argv[2]

#    app_name = 'quarkus'
#    workload = 'workload_50'

    filename = apps[app_name]
    with open(filename) as f:
        iterations = [iteration(line) for line in f]

    iterations = filter_out_workloads(workload, iterations)
    iteration0 = iterations[0]
    iterationN = ([iteration for iteration in iterations if iteration['status'] == 'TunedIteration'] or [None])[0]

    if iterationN is None:
        iterationN = iterations[-1]

    format_output_latex(app_name, workload,
            iteration0['production']['curr_config']['data'],
            iterationN['production']['curr_config']['data'])
