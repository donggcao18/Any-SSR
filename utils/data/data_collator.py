import logging
import torch
from transformers.data.data_collator import *

logger = logging.getLogger(__name__)


# Legacy ICL prompt glue (TRACE-era). Kept for backward compatibility only.
# HF instruction pool prompts are generated in `HFMultiTaskCodeDataset.get_prompt()`.
LEGACY_TASK_PROMPT = {
    "FOMC": "What is the monetary policy stance for the following text? A. dovish, B. hawkish, C. neutral. Choose one from A, B and C.\n",
    "C-STANCE": "判断以下文本对指定对象的态度，选择一项：A.支持，B.反对，C.中立。输出A，B或者C。\n",
    "ScienceQA": "Choose an answer for the following question and give your reasons.\n\n",
    "NumGLUE-cm": "Solve the following math problem.\n",
    "NumGLUE-ds": "Solve the following math problem.\n",
    "MeetingBank": "Write a summary of the following meeting transcripts.\n",
    "Py150": "Continue writing the code.\n",
    "20Minuten": "Provide a simplified version of the following paragraph in German.\n\n",
}

CONSTRAINED_PROMPT = "We will give you several examples and you should follow them to accomplish the task.\n Examples:\n"


def _is_hf_task(task: object) -> bool:
    return isinstance(task, str) and task.startswith("hf:")


def _strip_legacy_task_prefix(task: str, prompt: str) -> str:
    prefix = LEGACY_TASK_PROMPT.get(task)
    if not prefix:
        return prompt
    return prompt[len(prefix) :] if prompt.startswith(prefix) else prompt


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True  # ‘longest’
    max_prompt_len: Optional[int] = None
    max_ans_len: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 1
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    inference: bool = False
    demonstrations: Optional[Any] = None
    task: str = None

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        model_inputs = self.decoder_call(batch, self.return_tensors)

        return model_inputs

    # only support left padding for now
    def tokenize(self, sentence, cutoff_len, add_bos_token=True, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(
            sentence,
            truncation=True,
            max_length=cutoff_len,
            add_special_tokens=False,
            padding=False,
            return_tensors=None,
        )

        if (
                len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if (
                len(result["input_ids"]) < cutoff_len
                and add_bos_token
        ):
            result["input_ids"] = [self.tokenizer.bos_token_id] + result["input_ids"]
            result["attention_mask"] = [1] + result["attention_mask"]

        result["labels"] = result["input_ids"].copy()

        return result

    # Training: right padding. Inference: left padding.
    def decoder_call(self, batch, return_tensors):
        sources = []
        gts = []
        tokenized_sources = []
        actual_max_len = 0
        limit_len = self.max_prompt_len + self.max_ans_len if not self.inference else self.max_prompt_len

        pad_id = self.tokenizer.pad_token_id

        for instance in batch:
            instruction = instance['prompt']
            label = instance['answer']
            sources.append(instruction)
            gts.append(label)

            if not self.inference:
                # Wrap instruction in input/output template to steer generation format.
                formatted_prompt = f"input: {instruction}\noutput: "

                # Tokenize prompt and label separately — no BOS (Qwen has none),
                # EOS appended only at the end of the target.
                tokenize_prompt = self.tokenize(
                    formatted_prompt, self.max_prompt_len, add_bos_token=False, add_eos_token=False
                )
                tokenize_label = self.tokenize(
                    label, self.max_ans_len, add_bos_token=False, add_eos_token=True
                )
                prompt_len = len(tokenize_prompt["input_ids"])
                combined = {
                    "input_ids": tokenize_prompt["input_ids"] + tokenize_label["input_ids"],
                    "attention_mask": tokenize_prompt["attention_mask"] + tokenize_label["attention_mask"],
                    "labels": [-100] * prompt_len + tokenize_label["input_ids"].copy(),
                }
                tokenized_sources.append(combined)
            else:
                if self.demonstrations is not None:
                    # HF tasks: dataset has already produced instruction prompt via instruction pool.
                    # We only prepend demonstrations + constrained header; do NOT use legacy TASK_PROMT.
                    if _is_hf_task(self.task):
                        task_prompt = CONSTRAINED_PROMPT
                        for demonstration in self.demonstrations:
                            task_prompt += demonstration["prompt"]
                            task_prompt += demonstration["answer"] + "\n\n"
                        instruction = task_prompt + instruction
                    else:
                        # Legacy TRACE tasks
                        task_prefix = LEGACY_TASK_PROMPT.get(self.task, "")
                        task_prompt = task_prefix
                        if self.task != "MeetingBank":
                            task_prompt += CONSTRAINED_PROMPT
                        for demonstration in self.demonstrations:
                            if self.task == "Py150":
                                task_prompt += "Code:\n"
                            task_prompt += demonstration["prompt"]
                            task_prompt += demonstration["answer"] + "\n\n"

                        if self.task == "Py150":
                            task_prompt += "Code:\n"
                        if self.task != "Py150":
                            instruction = _strip_legacy_task_prefix(self.task, instruction)
                        instruction = task_prompt + instruction

                # Build formatted_prompt after demonstrations have been prepended to instruction.
                formatted_prompt = f"input: {instruction}\noutput: "

                # No BOS for inference either; left-pad with pad_token_id below.
                tokenize_source = self.tokenize(formatted_prompt, limit_len, add_bos_token=False, add_eos_token=False)
                tokenized_sources.append(tokenize_source)

            if len(tokenized_sources[-1]["input_ids"]) > actual_max_len:
                actual_max_len = len(tokenized_sources[-1]["input_ids"])

        actual_pad_len = (
            (actual_max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of
        )

        for idx in range(len(tokenized_sources)):
            pad_len = actual_pad_len - len(tokenized_sources[idx]["input_ids"])

            if not self.inference:
                # Left padding for training
                tokenized_sources[idx]["input_ids"] = (
                    [pad_id] * pad_len + tokenized_sources[idx]["input_ids"]
                )
                tokenized_sources[idx]["attention_mask"] = (
                    [0] * pad_len + tokenized_sources[idx]["attention_mask"]
                )
                tokenized_sources[idx]["labels"] = (
                    [-100] * pad_len + tokenized_sources[idx]["labels"]
                )
                assert (
                    len(tokenized_sources[idx]["input_ids"])
                    == len(tokenized_sources[idx]["attention_mask"])
                    == len(tokenized_sources[idx]["labels"])
                    == actual_pad_len
                )
            else:
                # Left padding for inference
                assert sum(tokenized_sources[idx]["attention_mask"]) == len(
                    tokenized_sources[idx]["input_ids"]
                )
                tokenized_sources[idx]["input_ids"] = (
                    [pad_id] * pad_len + tokenized_sources[idx]["input_ids"]
                )
                tokenized_sources[idx]["attention_mask"] = (
                    [0] * pad_len + tokenized_sources[idx]["attention_mask"]
                )

        model_inputs = {
            'input_ids': torch.tensor([source["input_ids"] for source in tokenized_sources]),
            'attention_mask': torch.tensor([source["attention_mask"] for source in tokenized_sources]),
        }

        if not self.inference:
            model_inputs['labels'] = torch.tensor([source["labels"] for source in tokenized_sources])

        model_inputs['sources'] = sources
        if self.inference:
            model_inputs['gts'] = gts

        return model_inputs


#   def decoder_call(self, batch, return_tensors):
#         sources = []
#         gts = []
#         tokenized_sources = []
#         actual_max_len = 0
#         limit_len = self.max_prompt_len + self.max_ans_len if not self.inference else self.max_prompt_len

#         pad_id = self.tokenizer.pad_token_id

#         for instance in batch:
#             instruction = instance['prompt']
#             label = instance['answer']
#             sources.append(instruction)
#             gts.append(label)

#             if not self.inference:
#                 # Wrap instruction in input/output template to steer generation format.
#                 formatted_prompt = f"input: {instruction}\noutput: "

#                 # Tokenize prompt and label separately — no BOS (Qwen has none),
#                 # EOS appended only at the end of the target.
#                 tokenize_prompt = self.tokenize(
#                     formatted_prompt, self.max_prompt_len, add_bos_token=False, add_eos_token=False
#                 )
#                 tokenize_label = self.tokenize(
#                     label, self.max_ans_len, add_bos_token=False, add_eos_token=True
#                 )
#                 prompt_len = len(tokenize_prompt["input_ids"])
#                 combined = {
#                     "input_ids": tokenize_prompt["input_ids"] + tokenize_label["input_ids"],
#                     "attention_mask": tokenize_prompt["attention_mask"] + tokenize_label["attention_mask"],
#                     "labels": [-100] * prompt_len + tokenize_label["input_ids"].copy(),
#                 }
#                 tokenized_sources.append(combined)
#             else:
#                 if self.demonstrations is not None:
#                     # HF tasks: dataset has already produced instruction prompt via instruction pool.
#                     # We only prepend demonstrations + constrained header; do NOT use legacy TASK_PROMT.
#                     if _is_hf_task(self.task):
#                         task_prompt = CONSTRAINED_PROMPT
#                         for demonstration in self.demonstrations:
#                             task_prompt += demonstration["prompt"]
#                             task_prompt += demonstration["answer"] + "\n\n"
#                         instruction = task_prompt + instruction
#                     else:
#                         # Legacy TRACE tasks
#                         task_prefix = LEGACY_TASK_PROMPT.get(self.task, "")
#                         task_prompt = task_prefix
#                         if self.task != "MeetingBank":
#                             task_prompt += CONSTRAINED_PROMPT
#                         for demonstration in self.demonstrations:
#                             if self.task == "Py150":
#                                 task_prompt += "Code:\n"
#                             task_prompt += demonstration["prompt"]
#                             task_prompt += demonstration["answer"] + "\n\n"

#                         if self.task == "Py150":
#                             task_prompt += "Code:\n"
#                         if self.task != "Py150":
#                             instruction = _strip_legacy_task_prefix(self.task, instruction)
#                         instruction = task_prompt + instruction

#                 # Build formatted_prompt after demonstrations have been prepended to instruction.
#                 formatted_prompt = f"input: {instruction}\noutput: "

#                 # No BOS for inference either; left-pad with pad_token_id below.
#                 tokenize_source = self.tokenize(formatted_prompt, limit_len, add_bos_token=False, add_eos_token=False)
#                 tokenized_sources.append(tokenize_source)

#             if len(tokenized_sources[-1]["input_ids"]) > actual_max_len:
#                 actual_max_len = len(tokenized_sources[-1]["input_ids"])

#         actual_pad_len = (
#             (actual_max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of
#         )

#         for idx in range(len(tokenized_sources)):
#             pad_len = actual_pad_len - len(tokenized_sources[idx]["input_ids"])

#             if not self.inference:
#                 # Right padding for training
#                 tokenized_sources[idx]["input_ids"] = (
#                     tokenized_sources[idx]["input_ids"] + [pad_id] * pad_len
#                 )
#                 tokenized_sources[idx]["attention_mask"] = (
#                     tokenized_sources[idx]["attention_mask"] + [0] * pad_len
#                 )
#                 tokenized_sources[idx]["labels"] = (
#                     tokenized_sources[idx]["labels"] + [-100] * pad_len
#                 )
#                 assert (
#                     len(tokenized_sources[idx]["input_ids"])
#                     == len(tokenized_sources[idx]["attention_mask"])
#                     == len(tokenized_sources[idx]["labels"])
#                     == actual_pad_len
#                 )
#             else:
#                 # Left padding for inference
#                 assert sum(tokenized_sources[idx]["attention_mask"]) == len(
#                     tokenized_sources[idx]["input_ids"]
#                 )
#                 tokenized_sources[idx]["input_ids"] = (
#                     [pad_id] * pad_len + tokenized_sources[idx]["input_ids"]
#                 )
#                 tokenized_sources[idx]["attention_mask"] = (
#                     [0] * pad_len + tokenized_sources[idx]["attention_mask"]
#                 )

#         model_inputs = {
#             'input_ids': torch.tensor([source["input_ids"] for source in tokenized_sources]),
#             'attention_mask': torch.tensor([source["attention_mask"] for source in tokenized_sources]),
#         }

#         if not self.inference:
#             model_inputs['labels'] = torch.tensor([source["labels"] for source in tokenized_sources])

#         model_inputs['sources'] = sources
#         if self.inference:
#             model_inputs['gts'] = gts

#         return model_inputs

