# This file defines the "fitness function" for the evolutionary process.
# It tells OpenEvolve how to measure the performance of each evolved program.
# The goal is to produce a score (or set of metrics) that the agent can use
# to decide which versions of the code are "better".

import time
import random
from openevolve.evaluate.evaluator import Evaluator, Program

class SortEvaluator(Evaluator):
    """
    Evaluates the performance of a sorting function.
    """
    def evaluate(self, program: Program):
        """
        Tests the sorting function for correctness and speed.
        
        A good evaluation function for this task would:
        1. Check if the list is actually sorted correctly (correctness).
        2. Measure the execution time (speed).
        
        The metrics dictionary is returned to the OpenEvolve controller.
        """
        try:
            # 1. Test for correctness
            test_list = [5, 1, 4, 2, 8]
            sorted_list = program.call("sort_list", test_list)
            
            correctness = 1.0 if sorted_list == sorted(test_list) else 0.0

            # 2. Test for performance (speed)
            # Use a larger list for a more meaningful speed test
            performance_list = [random.randint(0, 1000) for _ in range(500)]
            
            start_time = time.time()
            program.call("sort_list", performance_list)
            end_time = time.time()
            
            # We want to MINIMIZE execution time, so a lower value is better.
            execution_time = end_time - start_time
            
            # The metrics will be used by the evolutionary algorithm.
            # A higher correctness and lower time are better.
            program.metrics = {
                "correctness": correctness,
                "execution_time": execution_time
            }

        except Exception as e:
            # If the code is invalid or fails, give it the worst possible score.
            program.metrics = {
                "correctness": 0.0,
                "execution_time": float('inf')
            }
