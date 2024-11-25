def solve(setup, problem, setup_args, ):
    setup_params = setup(*setup_args)
    problem = problem(*setup_params)
    model = problem()[0]

    E, _, epsr = model(setup_params[1], setup_params[2], setup_params[4])
