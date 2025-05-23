def is_consistent(value,assignment):
    for i in assignment:
        if assignment[i]==value:
            return False
    return True
def select(variables,assignment):
    for var in variables:
        if var not in assignment:
            return var
    return None
def constraint(variables,assignment,domain):
    if len(variables)==len(assignment):
        return assignment
    variable=select(variables,assignment)
    for value in domain[variable]:
        if is_consistent(value,assignment):
            assignment[variable]=value
            result=constraint(variables,assignment,domain)
            if result:
                return result
            assignment.pop(variable)
    return None
domain={
    'X':[1,2,3],
    'Y':[1,2],
    'Z':[1,2]
    }
variables=['X','Y','Z']
print(constraint(variables,{},domain))
