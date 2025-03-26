def is_consistent(assignment, variable, value, constraints):
    # Check if the value assignment is consistent with the constraints
    for constraint in constraints:
        if not constraint(assignment, variable, value):
            return False
    return True

def backtrack(assignment, variables, domains, constraints):
    if len(assignment) == len(variables):
        return assignment  # All variables are assigned

    # Select an unassigned variable
    variable = select_unassigned_variable(variables, assignment)

    for value in domains[variable]:
        if is_consistent(assignment, variable, value, constraints):
            # Assign the value to the variable
            assignment[variable] = value
            # Recursively try to complete the assignment
            result = backtrack(assignment, variables, domains, constraints)
            if result:
                return result
            # If not successful, remove the assignment
            assignment.pop(variable)
    
    return None  # Trigger backtracking

def select_unassigned_variable(variables, assignment):
    for variable in variables:
        if variable not in assignment:
            return variable

# Example of constraints
def example_constraint(assignment, variable, value):
    # Example: All variables must have different values
    for var, val in assignment.items():
        if val == value:
            return False
    return True

if __name__ == "__main__":
    # Define variables
    variables = ['X1', 'X2', 'X3']

    # Define domains
    domains = {
        'X1': [1, 2, 3],
        'X2': [1, 2, 3],
        'X3': [1, 2, 3]
    }

    # Define constraints
    constraints = [example_constraint]

    # Solve CSP
    assignment = {}
    solution = backtrack(assignment, variables, domains, constraints)

    if solution:
        print("Solution found:", solution)
    else:
        print("No solution exists")
