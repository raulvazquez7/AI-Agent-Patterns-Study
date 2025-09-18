# This is the initial "DNA" for our evolutionary agent.
# OpenEvolve will iteratively modify this code to improve its performance
# based on the criteria defined in `evaluator.py`.

# The goal is to sort a list of numbers.
# This initial version is intentionally simple and inefficient.
# The LLM will try to evolve it into a more optimal solution.

def sort_list(numbers: list[int]) -> list[int]:
    """
    Sorts a list of integers using a basic bubble sort algorithm.
    This is a candidate for optimization.
    """
    n = len(numbers)
    # A simple bubble sort
    for i in range(n):
        for j in range(0, n - i - 1):
            if numbers[j] > numbers[j + 1]:
                numbers[j], numbers[j + 1] = numbers[j + 1], numbers[j]
    return numbers

