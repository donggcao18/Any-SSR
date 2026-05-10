"""Central HF task specs + instruction pools for Any-SSR HF adapter.

This is a lightweight port of the user's `task_info.py` idea but adapted for the
Any-SSR SFT pipeline (prompt+answer strings consumed by `DataCollator`).

Key rule: keep training methods unchanged; only data layer changes.
"""

from __future__ import annotations

from typing import Dict, List, TypedDict


TASK_LIST: List[str] = [
    "CONCODE",
    "CodeTrans",
    "CodeSearchNet",
    "BFP",
    "KodCode",
    "RunBugRun",
    "TheVault_Csharp",
    "CoST"
]


class TaskSpec(TypedDict, total=False):
    dataset_name: str
    text_key: str
    label_key: str
    task_type: str
    language: str
    source_lang: str
    target_lang: str


TASK_SPECS: Dict[str, TaskSpec] = {
    "CONCODE": {
        "dataset_name": "AhmedSSoliman/CodeXGLUE-CONCODE",
        "text_key": "nl",
        "label_key": "code",
        "task_type": "code_generation",
        "language": "Java",
    },
    "CodeTrans": {
        "dataset_name": "CM/codexglue_codetrans",
        "text_key": "java",
        "label_key": "cs",
        "task_type": "code_translation",
        "source_lang": "Java",
        "target_lang": "C#",
    },
    "BFP": {
        "dataset_name": "ayeshgk/code_x_glue_cc_code_refinement_annotated",
        "text_key": "buggy",
        "label_key": "fixed",
        "task_type": "code_refinement",
        "language": "Java",
    },
    "KodCode": {
        "dataset_name": "KodCode/KodCode-V1-SFT-R1",
        "text_key": "question",
        "label_key": "solution",
        "task_type": "code_generation",
        "language": "Python",
    },
    "RunBugRun": {
        "dataset_name": "ASSERT-KTH/RunBugRun-Final",
        "text_key": "buggy_code",
        "label_key": "fixed_code",
        "task_type": "code_refinement",
        "language": "Ruby",
    },
    "CoST": {
        "dataset_name": "dongg18/CoST",
        "text_key": "lang1",
        "label_key": "lang2",
        "task_type": "code_translation",
        "source_lang": "C++",
        "target_lang": "C#",
    },
    "CodeSearchNet": {
        "dataset_name": "semeru/code-text-ruby",
        "text_key": "code",
        "label_key": "docstring",
        "task_type": "code_summarization",
        "language": "Ruby",
    },
    "TheVault_Csharp": {
        "dataset_name": "Fsoft-AIC/the-vault-function",
        "text_key": "code",
        "label_key": "docstring",
        "task_type": "code_summarization",
        "language": "C#",
    },
}


TRAIN_ONLY_TASKS = {
    # Split sizes for turning train-only datasets into train/validation/test
    "KodCode": {"val": 5000, "test": 5000},
    "RunBugRun": {"val": 972, "test": 1000},
}


INSTRUCTION_POOL: Dict[str, List[str]] = {
    "code_summarization": [
        "Summarize the following {language} code:\n{code}",
        "Summarize this code:\n{code}",
        "What does this {language} code do?\n{code}",
        "What does the following code do?\n{code}",
        "Describe this code:\n{code}",
        "Explain what the following {language} code does:\n{code}",
        "Explain this code to me:\n{code}",
        "Give a brief explanation of the following code:\n{code}",
        "Give me a short description of what this {language} code does:\n{code}",
        "Write a summary for the {language} code below:\n{code}",
        "Provide a concise summary of the following {language} code snippet:\n{code}",
        "Read the code below and provide a brief description of its functionality:\n{code}",
        "What does this code snippet do?\n{code}",
        "I have this {language} code, what does it do?\n{code}",
        "Look at this code and tell me what it's doing:\n{code}",
        "What is the purpose of the following {language} function?\n{code}",
        "Give me a high-level overview of what this code does:\n{code}",
        "Review this {language} code and give a summary of its functionality:\n{code}",
        "This is a {language} function, summarize what it does for documentation:\n{code}",
    ],
    "code_generation": [
        "Write a {language} function that {description}",
        "Implement a {language} function that {description}",
        "Write {language} code that {description}",
        "Generate a {language} program that {description}",
        "Create a {language} function to {description}",
        "Write a {language} script that {description}",
        "Implement the following in {language}: {description}",
        "Here is a programming problem, solve it in {language}:\n{description}",
        "Write a solution in {language} for the following:\n{description}",
        "I need a {language} function that {description}",
        "I want to {description}. Write the {language} code for it.",
        "How do I {description} in {language}? Write the code.",
        "Code a {language} function that {description}",
        "{description}\nSolve this in {language}.",
        "Build a {language} function that {description}",
        "Give me {language} code that {description}",
        "Write me a {language} program that {description}",
        "Here's what I need: {description}\nWrite it in {language}.",
    ],
    "code_translation": [
        "Translate the following {source_lang} code to {target_lang}:\n{code}",
        "Convert this {source_lang} code to {target_lang}:\n{code}",
        "Rewrite the following code in {target_lang}:\n{code}",
        "Rewrite this {source_lang} code in {target_lang}:\n{code}",
        "Port the following {source_lang} code to {target_lang}:\n{code}",
        "Convert this code from {source_lang} to {target_lang}:\n{code}",
        "Here is some {source_lang} code. Translate it to {target_lang}:\n{code}",
        "I have this {source_lang} code, convert it to {target_lang}:\n{code}",
        "Migrate this {source_lang} code to {target_lang}:\n{code}",
        "I need this code in {target_lang} instead of {source_lang}:\n{code}",
        "What is the {target_lang} equivalent of this {source_lang} code?\n{code}",
        "How would this {source_lang} code look in {target_lang}?\n{code}",
        "Turn this {source_lang} snippet into {target_lang}:\n{code}",
        "Take this {source_lang} function and write it in {target_lang}:\n{code}",
        "Transform the following {source_lang} code into {target_lang}:\n{code}",
        "Write the {target_lang} version of this {source_lang} code:\n{code}",
        "This code is in {source_lang}, rewrite it in {target_lang}:\n{code}",
        "I want to switch from {source_lang} to {target_lang} for this function:\n{code}",
        "Give me the {target_lang} translation of the following {source_lang} code:\n{code}",
        "Make this {source_lang} code work in {target_lang}:\n{code}",
    ],
    "code_refinement": [
        "Refine the following {language} code:\n{code}",
        "Improve this {language} code:\n{code}",
        "Fix the bugs in the following {language} code:\n{code}",
        "Optimize the following {language} code:\n{code}",
        "Find and fix any issues in this {language} code:\n{code}",
        "Debug the following {language} code:\n{code}",
        "This {language} code has bugs, fix them:\n{code}",
        "Make this {language} code better:\n{code}",
        "Improve the quality of the following {language} code:\n{code}",
        "There's something wrong with this code, fix it:\n{code}",
        "This code doesn't work correctly, fix it:\n{code}",
        "Rewrite this {language} code to be cleaner and more efficient:\n{code}",
        "Review and improve the following {language} code:\n{code}",
        "I wrote this {language} code but it has issues, fix it:\n{code}",
        "Identify and fix the problems in this {language} code:\n{code}",
        "This {language} code is buggy, correct it:\n{code}",
        "Refactor this {language} code to improve readability and performance:\n{code}",
        "What's wrong with this {language} code? Fix it:\n{code}",
    ],
    "test_generation": [
        "Write unit tests for the following {language} code:\n{code}",
        "Generate tests for this {language} function:\n{code}",
        "Write test cases for the following {language} code:\n{code}",
        "Create unit tests for this {language} code:\n{code}",
        "Generate unit tests for the following {language} function:\n{code}",
        "Write a test suite for this {language} code:\n{code}",
        "I need unit tests for this {language} function:\n{code}",
        "Test this {language} code:\n{code}",
        "Generate test cases to verify this {language} code:\n{code}",
        "Write tests to validate the following {language} function:\n{code}",
        "This is a {language} function, write tests for it:\n{code}",
        "Here's my {language} code, write unit tests for it:\n{code}",
        "Cover this {language} function with unit tests:\n{code}",
        "Write comprehensive tests for the following {language} code:\n{code}",
        "I want to test this {language} function, write the test cases:\n{code}",
        "Generate edge case tests for the following {language} code:\n{code}",
        "What tests should I write for this {language} function? Write them:\n{code}",
        "Make sure this {language} code is well-tested, write the tests:\n{code}",
        "Write both normal and edge case tests for this {language} code:\n{code}",
        "Add tests for the following {language} function:\n{code}",
    ],
}


INSTRUCTION_SPLIT_POLICY = {
    "train": {"pool_scope": "head_fraction", "fraction": 0.75},
    "validation": {"pool_scope": "head_fraction", "fraction": 0.75},
    "dev": {"pool_scope": "head_fraction", "fraction": 0.75},
    "test": {"pool_scope": "full", "fraction": 1.0},
}
