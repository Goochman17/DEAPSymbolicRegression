import operator
import math
import random
import numpy
import networkx as nx
import sympy
import copy
import matplotlib.pyplot as plt
import pandas as pd
from deap import algorithms, base, creator, tools, gp

data1 = pd.read_csv(
    "/Users/k/Desktop/4th_Year/First_Semester/CSCI 340 - Evolutionary Computation/Asn 2/someData/d1.csv"
)
data2 = pd.read_csv(
    "/Users/k/Desktop/4th_Year/First_Semester/CSCI 340 - Evolutionary Computation/Asn 2/someData/d2.csv"
)
data3 = pd.read_csv(
    "/Users/k/Desktop/4th_Year/First_Semester/CSCI 340 - Evolutionary Computation/Asn 2/someData/d3.csv"
)
data4 = pd.read_csv(
    "/Users/k/Desktop/4th_Year/First_Semester/CSCI 340 - Evolutionary Computation/Asn 2/someData/d4.csv"
)
data5 = pd.read_csv(
    "/Users/k/Desktop/4th_Year/First_Semester/CSCI 340 - Evolutionary Computation/Asn 2/someData/d5.csv"
)
data6 = pd.read_csv(
    "/Users/k/Desktop/4th_Year/First_Semester/CSCI 340 - Evolutionary Computation/Asn 2/someData/d6.csv"
)
data7 = pd.read_csv(
    "/Users/k/Desktop/4th_Year/First_Semester/CSCI 340 - Evolutionary Computation/Asn 2/someData/d7.csv"
)
test_data = pd.read_csv("/Users/k/Desktop/test.csv")

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


pset = gp.PrimitiveSet("MAIN", arity=1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="y")
pset.renameArguments(ARG2="z")


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function
    sqerrors = []
    for i in range(len(points)):
        sqerrors.append((float(points.iloc[i, -1]) - func(*points.iloc[i, :-1])) ** 2)
    return (math.fsum(sqerrors) / len(points),)


toolbox.register("evaluate", evalSymbReg, points=test_data)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", numpy.mean)
mstats.register("std", numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)

pop = toolbox.population(n=300)
hof = tools.HallOfFame(1)
pop, log = algorithms.eaSimple(
    pop, toolbox, 0.5, 0.1, 40, stats=mstats, halloffame=hof, verbose=True
)

expr = toolbox.individual()
nodes, edges, labels = gp.graph(expr)
g = nx.Graph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="dot")
nx.draw_networkx_nodes(g, pos)
nx.draw_networkx_edges(g, pos)
nx.draw_networkx_labels(g, pos, labels)
plt.show()


def convert_inverse_prim(prim, args):
    """
    Convert inverse prims according to:
    [Dd]iv(a,b) -> Mul[a, 1/b]
    [Ss]ub(a,b) -> Add[a, -b]
    We achieve this by overwriting the corresponding format method of the sub and div prim.
    """
    prim = copy.copy(prim)
    # prim.name = re.sub(r'([A-Z])', lambda pat: pat.group(1).lower(), prim.name)    # lower all capital letters

    converter = {
        "sub": lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        "protectedDiv": lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        "mul": lambda *args_: "Mul({},{})".format(*args_),
        "add": lambda *args_: "Add({},{})".format(*args_),
        "neg": lambda *args_: "Mul(-1,{})".format(*args_),
    }
    prim_formatter = converter.get(prim.name, prim.format)

    return prim_formatter(*args)


def stringify_for_sympy(f):
    """Return the expression in a human readable string."""
    string = ""
    stack = []
    for node in f:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = convert_inverse_prim(prim, args)
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(string)
    return string


print(hof[0])
function = sympy.simplify(stringify_for_sympy(hof[0]))
function = str(function)
function = function.replace("sin", "numpy.sin")
function = function.replace("cos", "numpy.cos")
function = function.replace("sqrt", "numpy.sqrt")
function = function.replace("pi", "numpy.pi")
function = function.replace("neg", "-")
print(function)

results = []
for i in range(len(test_data)):
    x = test_data.iloc[i, 0]
    y = test_data.iloc[i, 1]
    results.append(eval(function))

plt.scatter(test_data.iloc[:, 0], test_data.iloc[:, -1], label="Actual")
plt.scatter(test_data.iloc[:, 0], results, label="Predicted")
plt.legend(loc="upper left")
plt.show()
