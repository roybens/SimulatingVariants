from neuron import h, gui
import re
import numpy as np

# Reads names of parameters for the model provided by the mod file and changes parameters to specified
def ChangeParameters(cell, NEW_PARAMS, MOD_FILE):
    param_list, suffix, neat_data = ReadParameterNames(MOD_FILE)

    global_vars = []
    range_vars = []

    for line in neat_data['NEURON']:
        if re.match('^GLOBAL', line):
            line = re.split('^GLOBAL\s+', line)[1]
            global_vars.extend(re.split('\s*,\s*', line))
        elif re.match('^RANGE', line):
            line = re.split('^RANGE\s+', line)[1]
            range_vars.extend(re.split('\s*,\s*', line))

    global_vars = [s + "_" + suffix for s in global_vars]
    range_vars = [s + "_" + suffix for s in range_vars]


    for variable in NEW_PARAMS.keys():
        try:
            param_list[variable] = NEW_PARAMS.get(variable)
            if variable in global_vars:
                exec("cell.h." + variable + "=" + str(NEW_PARAMS.get(variable)))
            elif variable in range_vars:
                exec("cell.soma." + variable + "=" + str(NEW_PARAMS.get(variable)))
        except:
            print("Variable " + variable + " is not a parameter of " + suffix)

    return cell



# Reads the names and inital values of parameters for the model
def ReadParameterNames(MOD_FILE):
    param_list = {}

    data = open(MOD_FILE, 'r').readlines()
    for n in np.arange(len(data)):
        data[n] = data[n].rstrip()
        data[n] = re.sub('\t*', '', data[n])
        data[n] = re.sub('^\s*', '', data[n])
    if re.match('\r\n', data[n]):
        data[n].remove()

    neat_data = {}
    category = ""
    cat_data = []

    for line in data:
        if re.match("[A-Z]+\s*\{", line):
            category = re.search("[A-Z]+", line).group(0)
        elif re.match("\s*\}", line):
            neat_data[category] = cat_data
            cat_data = []
            category = ""
        else:
            if category != "" and line != "":
                cat_data.append(line)

    suffix = re.sub('SUFFIX[\s]*', '', neat_data['NEURON'][0])

    for param in neat_data['PARAMETER']:
        try:
            param_list[re.search("^[A-Za-z0-9_]+", param).group(0) + "_" + suffix] \
                = float(re.search("(=\s*)([-]?[\d]*\.?\d+(?:e[-+]\d*)?)", param).group(2))
        except:
            continue

    return param_list, suffix, neat_data

