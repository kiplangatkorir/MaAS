import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple

from maas.ext.maas.scripts.optimized.HumanEval.train.template.operator_an import *
from maas.ext.maas.scripts.optimized.HumanEval.train.template.op_prompt import *
from maas.ext.maas.scripts.utils import extract_test_cases_from_jsonl, test_case_2_test_function
from maas.actions.action_node import ActionNode
from maas.llm import LLM
from maas.logs import logger
import re


class Operator:
    """
    Base class for all operators in the MaAS framework.
    
    Operators are the fundamental building blocks that encapsulate specific LLM-based operations.
    Each operator takes an LLM instance and performs a specific task by generating prompts,
    processing LLM responses, and returning structured results.
    
    The base class provides common functionality for all operators, including the _fill_node method
    which handles the interaction with the ActionNode system for prompt filling and response parsing.
    """
    def __init__(self, llm: LLM, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        fill_kwargs = {"context": prompt, "llm": self.llm}
        if mode:
            fill_kwargs["mode"] = mode
        fill_kwargs.update(extra_kwargs)
        node = await ActionNode.from_pydantic(op_class).fill(**fill_kwargs)
        return node.instruct_content.model_dump()

class CustomCodeGenerate(Operator):
    """
    Operator for generating code solutions with custom instructions.
    
    This operator allows for customized code generation by accepting an explicit instruction
    parameter that is prepended to the problem statement. This provides flexibility in how
    the LLM is prompted to generate code solutions.
    
    The operator uses the GenerateOp action node defined in operator_an.py and the code_fill
    mode to extract structured code responses from the LLM.
    
    Args:
        llm (LLM): The language model to use for code generation
        name (str): The name of the operator, defaults to "CustomCodeGenerate"
    """
    def __init__(self, llm: LLM, name: str = "CustomCodeGenerate"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction):
        """
        Generate code based on a problem statement with custom instructions.
        
        Args:
            problem (str): The problem statement/description
            entry_point (str): The name of the function to be implemented
            instruction (str): Custom instructions to guide the code generation
            
        Returns:
            dict: The generated code solution
        """
        prompt = instruction + problem
        response = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        return response

class Generate(Operator):
    """
    Standard code generation operator.
    
    This operator generates code solutions for programming problems by combining
    the provided instruction with the problem statement. It uses the GenerateOp
    action node from operator_an.py to structure the interaction with the LLM.
    
    The prompt templates used by this operator are defined in op_prompt.py.
    
    Args:
        llm (LLM): The language model to use for code generation
        name (str): The name of the operator, defaults to "Generate"
    """
    def __init__(self, llm: LLM, name: str = "Generate"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction):
        """
        Generate code based on a problem statement.
        
        Args:
            problem (str): The problem statement/description
            entry_point (str): The name of the function to be implemented
            instruction (str): Instructions to guide the code generation
            
        Returns:
            dict: The generated code solution
        """
        prompt = instruction + problem
        response = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        return response
    
class GenerateCoT(Operator):
    """
    Chain-of-Thought code generation operator.
    
    This operator implements the Chain-of-Thought (CoT) approach for code generation,
    which encourages the LLM to reason step-by-step before providing a solution.
    While the implementation is similar to the standard Generate operator, the
    instruction parameter typically contains CoT-specific prompting.
    
    The CoT approach has been shown to improve reasoning capabilities in LLMs by
    breaking down complex problems into intermediate steps.
    
    The prompt templates used by this operator are defined in op_prompt.py.
    
    Args:
        llm (LLM): The language model to use for code generation
        name (str): The name of the operator, defaults to "GenerateCoT"
    """
    def __init__(self, llm: LLM, name: str = "GenerateCoT"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction):
        """
        Generate code using Chain-of-Thought reasoning.
        
        Args:
            problem (str): The problem statement/description
            entry_point (str): The name of the function to be implemented
            instruction (str): CoT-specific instructions to guide the reasoning process
            
        Returns:
            dict: The generated code solution after CoT reasoning
        """
        prompt = instruction + problem
        response = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        return response

class MultiGenerateCoT(Operator):
    """
    Multiple sampling Chain-of-Thought code generation operator.
    
    This operator extends the Chain-of-Thought approach by generating multiple
    solutions for the same problem. It makes three separate calls to the LLM
    with the same prompt, leveraging the inherent randomness in LLM sampling
    to produce diverse solutions. These multiple solutions can later be used
    with ensemble methods like ScEnsemble to select the best solution.
    
    The multiple sampling approach increases the likelihood of finding a correct
    solution by exploring different paths in the LLM's solution space.
    
    The prompt templates used by this operator are defined in op_prompt.py.
    
    Args:
        llm (LLM): The language model to use for code generation
        name (str): The name of the operator, defaults to "MultiGenerateCoT"
    """
    def __init__(self, llm: LLM, name: str = "MultiGenerateCoT"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction):
        """
        Generate multiple code solutions using Chain-of-Thought reasoning.
        
        This method makes three separate calls to the LLM with identical inputs
        to generate three different solutions due to the sampling randomness.
        
        Args:
            problem (str): The problem statement/description
            entry_point (str): The name of the function to be implemented
            instruction (str): CoT-specific instructions to guide the reasoning process
            
        Returns:
            dict: A dictionary containing a list of three generated code solutions
        """
        prompt = instruction + problem
        
        response1 = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        response2 = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        response3 = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        
        return {"response": [response1, response2, response3]}
    
class ScEnsemble(Operator):
    """
    Self-Consistency Ensemble operator for selecting the best solution from multiple candidates.
    
    This operator implements the Self-Consistency (SC) approach described in the following papers:
    - "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
      (https://arxiv.org/abs/2203.11171)
    - "Universal Self-Consistency for Large Language Model Generation"
      (https://arxiv.org/abs/2311.17311)
    
    The SC approach works by generating multiple solutions (typically using MultiGenerateCoT)
    and then asking the LLM to evaluate and select the best solution among them. This leverages
    the LLM's ability to critique solutions, which is often more reliable than its ability
    to generate perfect solutions on the first try.
    
    The operator uses the SC_ENSEMBLE_PROMPT template from op_prompt.py and the ScEnsembleOp
    action node from operator_an.py to structure the interaction with the LLM.
    
    Args:
        llm (LLM): The language model to use for ensemble selection
        name (str): The name of the operator, defaults to "ScEnsemble"
    """
    def __init__(self, llm: LLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    async def __call__(self, solutions: List[str], problem: str):
        """
        Select the best solution from multiple candidates using Self-Consistency.
        
        This method formats multiple solutions with letter labels (A, B, C, etc.),
        presents them to the LLM along with the original problem, and asks the LLM
        to select the best solution. The selection is returned as the final answer.
        
        Args:
            solutions (List[str]): List of candidate solutions to evaluate
            problem (str): The original problem statement/description
            
        Returns:
            dict: A dictionary containing the selected best solution
        """
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        # SC_ENSEMBLE_PROMPT is defined in op_prompt.py
        prompt = SC_ENSEMBLE_PROMPT.format(problem=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}

class Test(Operator):
    """
    Test operator for evaluating and refining code solutions.
    
    This operator executes code solutions against test cases and provides feedback
    for refinement. It implements a test-and-refine loop that:
    1. Executes the solution against test cases from the HumanEval dataset
    2. If tests fail, uses the LLM to reflect on the errors and generate an improved solution
    3. Repeats the process for a specified number of iterations or until all tests pass
    
    The operator uses utility functions from maas.ext.maas.scripts.utils to extract
    and format test cases, and the ReflectionTestOp action node from operator_an.py
    to structure the reflection and refinement process.
    
    The REFLECTION_ON_PUBLIC_TEST_PROMPT template from op_prompt.py is used to guide
    the LLM's reflection process.
    
    Args:
        llm (LLM): The language model to use for reflection and refinement
        name (str): The name of the operator, defaults to "Test"
    """
    def __init__(self, llm: LLM, name: str = "Test"):
        super().__init__(llm, name)

    def exec_code(self, solution, entry_point):
        """
        Execute a code solution against test cases and collect error information.
        
        This method:
        1. Extracts test cases for the specified entry point from the HumanEval dataset
        2. Converts each test case into executable test code
        3. Executes the tests and captures any errors or assertion failures
        4. Returns detailed error information or "no error" if all tests pass
        
        Args:
            solution (str): The code solution to test
            entry_point (str): The name of the function being tested
            
        Returns:
            str or list or dict: "no error" if all tests pass, a list of test failure details,
                                or a dictionary with execution error information
        """
        # Extract test cases from the HumanEval dataset for the specified entry point
        test_cases = extract_test_cases_from_jsonl(entry_point, dataset="HumanEval")
                
        fail_cases = []
        for test_case in test_cases:
            # Convert the test case to executable Python code
            test_code = test_case_2_test_function(solution, test_case, entry_point)
            try:
                # Execute the test code
                exec(test_code, globals())
            except AssertionError as e:
                # Capture assertion failures (test failures)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                with open("tester.txt", "a") as f:
                    f.write("test_error of " + entry_point + "\n")
                error_infomation = {
                    "test_fail_case": {
                        "test_case": test_case,
                        "error_type": "AssertionError",
                        "error_message": str(e),
                        "traceback": tb_str,
                    }
                }
                fail_cases.append(error_infomation)
            except Exception as e:
                # Capture execution errors (syntax errors, runtime errors, etc.)
                with open("tester.txt", "a") as f:
                    f.write(entry_point + " " + str(e) + "\n")
                return {"exec_fail_case": str(e)}
        
        # Return test results
        if fail_cases != []:
            return fail_cases
        else:
            return "no error"

    async def __call__(
        self, problem, solution, entry_point, test_loop: int = 3
    ):
        """
        Test a solution and refine it through multiple iterations if needed.
        
        This method implements a test-and-refine loop that:
        1. Tests the solution against test cases
        2. If tests pass, returns the solution as correct
        3. If tests fail, uses the LLM to reflect on the errors and generate an improved solution
        4. Repeats for up to test_loop iterations
        
        The reflection process uses the REFLECTION_ON_PUBLIC_TEST_PROMPT template from op_prompt.py
        and the ReflectionTestOp action node to guide the LLM in understanding and fixing errors.
        
        Args:
            problem (str): The problem statement/description
            solution (str): The initial code solution to test
            entry_point (str): The name of the function being tested
            test_loop (int): Maximum number of test-and-refine iterations, defaults to 3
            
        Returns:
            dict: A dictionary containing the result (True/False) and the final solution
        """
        for _ in range(test_loop):
            result = self.exec_code(solution, entry_point)
            if result == "no error":
                return {"result": True, "solution": solution}
            elif "exec_fail_case" in result:
                result = result["exec_fail_case"]
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass=f"executed unsuccessfully, error: \n {result}",
                    test_fail="executed unsucessfully",
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response["reflection_and_solution"]
            else:
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass="executed successfully",
                    test_fail=result,
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response["reflection_and_solution"]
        
        result = self.exec_code(solution, entry_point)
        if result == "no error":
            return {"result": True, "solution": solution}
        else:
            return {"result": False, "solution": solution}
        
class SelfRefine(Operator):
    """
    Self-Refinement operator for improving code solutions without test execution.
    
    This operator implements a self-refinement approach where the LLM is asked to
    critique and improve its own solution. Unlike the Test operator, SelfRefine
    doesn't execute the code against test cases but relies solely on the LLM's
    ability to identify and fix potential issues in the code.
    
    The operator uses the SELFREFINE_PROMPT template from op_prompt.py and the
    SelfRefineOp action node from operator_an.py to structure the interaction
    with the LLM.
    
    Args:
        llm (LLM): The language model to use for self-refinement
        name (str): The name of the operator, defaults to "SelfRefine"
    """
    def __init__(self, llm: LLM, name: str = "SelfRefine"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution):
        """
        Refine a code solution without test execution.
        
        This method asks the LLM to critique and improve the given solution
        based on the problem statement alone, without executing any tests.
        
        Args:
            problem (str): The problem statement/description
            solution (str): The code solution to refine
            
        Returns:
            dict: The refined code solution
        """
        # SELFREFINE_PROMPT is defined in op_prompt.py
        prompt = SELFREFINE_PROMPT.format(problem=problem, solution=solution)
        response = await self._fill_node(SelfRefineOp, prompt, mode="code_fill")
        return response
    
class EarlyStop(Operator):
    """
    Early stopping operator for terminating processing pipelines.
    
    This operator is a placeholder for implementing early stopping logic in
    processing pipelines. Early stopping can be useful to terminate processing
    when certain conditions are met, such as when a solution passes all tests
    or when a maximum number of refinement iterations has been reached.
    
    Currently, this operator is not fully implemented and will raise NotImplementedError
    when called.
    
    Args:
        llm (LLM): The language model instance (not used in current implementation)
        name (str): The name of the operator, defaults to "EarlyStop"
    """
    def __init__(self, llm: LLM, name: str = "EarlyStop"):
        super().__init__(llm, name)

    async def __call__(self):
        """
        Placeholder for early stopping logic.
        
        This method is not yet implemented and will raise NotImplementedError when called.
        
        Returns:
            NotImplementedError: Always raises this exception
        """
        return NotImplementedError
