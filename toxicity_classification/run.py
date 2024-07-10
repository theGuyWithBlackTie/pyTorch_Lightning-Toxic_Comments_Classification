import argparse

from params import experiment_params

if __name__ == "__main__":
    '''
    Reading the command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", help="Optimizer to use", default="adam")
    parser.add_argument("--scheduler", help="Scheduler to use", default="cosine-annealing")
    args = parser.parse_args()
    print(args)
    experiment_params.setdefault("optimizer", args.optimizer)
    experiment_params.setdefault("scheduler", args.scheduler)


