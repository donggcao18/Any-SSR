import torch
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
import json
import os
import time
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds, eval_20Minuten # to be continued
from evaluator.compute_metrics import compute_metrics, DATASET_TO_OUTPUT_LANG
from transformers import GenerationConfig

class CL_Base_Model:
    def __init__(self,
                 model,
                 tokenizer,
                 optimizer,
                 train_task_list,
                 eval_task_list,
                 test_task_list,
                 args):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_task_list = train_task_list
        self.eval_task_list = eval_task_list
        self.test_task_list = test_task_list
        self.args = args
        self.generation_config = GenerationConfig(
            do_sample=self.args.do_sample,
            temperature=self.args.temperature if self.args.do_sample else None,
            top_p=self.args.top_p if self.args.do_sample else None,
            repetition_penalty=self.args.repetition_penalty,
        )

    def perplexity_evaluation(self, eval_dataloader, device):
        # 验证集上测困惑度
        self.model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            del batch['sources']
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = self.model(**batch, use_cache=False)
            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    def _task_eval_from_predictions(self, task, sources_sequences, predicted_sequences, ground_truths):
        if task in ['CodeSearchNet', 'TheVault_Csharp']:
            calc_codebleu = False
        else:
            calc_codebleu = True
        return compute_metrics(predicted_sequences, ground_truths, calc_codebleu=calc_codebleu, language=DATASET_TO_OUTPUT_LANG.get(task, None))

    def task_generation_evaluation(self, task, test_dataloader, device, max_ans_len=None, return_predictions=False):
        self.model.eval()
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []

        if max_ans_len is None:
            max_ans_len = getattr(self.args, "max_ans_len", 256)

        is_executable = getattr(self.args, "benchmark", "non-executable") != "non-executable"
        if is_executable:
            return_predictions = True
            num_return_sequences = int(getattr(self.args, "num_return_sequences", 1))
            top_k = int(getattr(self.args, "top_k", 0))
            generation_kwargs = self.generation_config.to_dict()
            generation_kwargs.update({
                "num_return_sequences": num_return_sequences,
                "top_k": top_k,
            })
            generation_config = GenerationConfig(**generation_kwargs)
        else:
            num_return_sequences = 1
            generation_config = self.generation_config

        progress_bar = tqdm(total=len(test_dataloader), leave=True, disable=(self.args.global_rank != 0))
        for step, batch in enumerate(test_dataloader):
            sources_sequences += batch['sources']
            if 'gts' in batch:
                ground_truths += batch['gts']
                del batch['gts']
            elif 'labels' in batch:
                label_tensor = batch['labels']
                for row in label_tensor:
                    valid_ids = row[row != -100].detach().cpu().tolist()
                    ground_truths.append(
                        self.tokenizer.decode(valid_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    )
                del batch['labels']
            else:
                ground_truths += [''] * len(batch['sources'])

            del batch['sources']
            batch = to_device(batch, device)
            prompt_len = batch['input_ids'].shape[1]

            with torch.no_grad():
                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = self.tokenizer.eos_token_id

                generate_ids = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=max_ans_len,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    use_cache=True,
                )

            sequences = self.tokenizer.batch_decode(
                generate_ids[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            if is_executable and num_return_sequences > 1:
                batch_preds = [
                    sequences[i:i + num_return_sequences]
                    for i in range(0, len(sequences), num_return_sequences)
                ]
                predicted_sequences.extend(batch_preds)
            else:
                predicted_sequences += sequences

            if self.args.global_rank == 0:
                progress_bar.update(1)
                description = f"Test step {step}"
                progress_bar.set_description(description, refresh=False)

        metrics = {} if is_executable else self._task_eval_from_predictions(
            task, sources_sequences, predicted_sequences, ground_truths
        )
        if return_predictions:
            prediction_rows = [
                {
                    "source": source,
                    "ground-truth": gt,
                    "prediction": pred,
                }
                for source, gt, pred in zip(sources_sequences, ground_truths, predicted_sequences)
            ]
            return metrics, prediction_rows
        return metrics

    def _resolve_max_ans_len(self, task_idx):
        max_ans_len = getattr(self.args, "max_ans_len", 256)
        if isinstance(max_ans_len, (list, tuple)):
            if len(max_ans_len) == 0:
                return 256
            if len(max_ans_len) == 1:
                return int(max_ans_len[0])
            return int(max_ans_len[task_idx])
        return int(max_ans_len)

    def test_all_tasks_and_save_predictions(self):
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)

        prediction_root = os.path.join(self.args.output_dir or ".", "predictions", f"final-{self.__class__.__name__}")
        if self.args.global_rank == 0:
            os.makedirs(prediction_root, exist_ok=True)

        if getattr(self.args, "benchmark", "non-executable") == "non-executable":
            final_metrics = {}
            for task_idx, (task_name, test_dataloader) in enumerate(self.test_task_list.items()):
                print_rank_0(
                    f"***** Final testing on task {task_name} after continual training *****",
                    self.args.global_rank,
                )
                test_result, prediction_rows = self.task_generation_evaluation(
                    task_name,
                    test_dataloader,
                    device,
                    max_ans_len=self._resolve_max_ans_len(task_idx),
                    return_predictions=True,
                )
                final_metrics[task_name] = test_result
                print_rank_0(f"[final-test task={task_name}] result: {test_result}", self.args.global_rank)

                if self.args.global_rank == 0:
                    safe_task_name = str(task_name).replace("/", "_").replace(":", "_")
                    prediction_file = os.path.join(prediction_root, f"{task_idx}_{safe_task_name}.json")
                    with open(prediction_file, "w", encoding="utf-8") as f:
                        json.dump(prediction_rows, f, ensure_ascii=False, indent=2)
                    print_rank_0(f"Saved final-test predictions to {prediction_file}", self.args.global_rank)

            if self.args.global_rank == 0:
                metrics_file = os.path.join(prediction_root, "metrics_summary.json")
                with open(metrics_file, "w", encoding="utf-8") as f:
                    json.dump(final_metrics, f, ensure_ascii=False, indent=2)
                print_rank_0(f"Saved final-test metrics to {metrics_file}", self.args.global_rank)
        else:
            for task_idx, (task_name, test_dataloader) in enumerate(self.test_task_list.items()):
                print_rank_0(
                    f"***** Final testing on task {task_name} after continual training *****",
                    self.args.global_rank,
                )
                _, prediction_rows = self.task_generation_evaluation(
                    task_name,
                    test_dataloader,
                    device,
                    max_ans_len=self._resolve_max_ans_len(task_idx),
                    return_predictions=True,
                )

                if self.args.global_rank == 0:
                    safe_task_name = str(task_name).replace("/", "_").replace(":", "_")
                    prediction_file = os.path.join(prediction_root, f"{task_idx}_{safe_task_name}.json")
                    with open(prediction_file, "w", encoding="utf-8") as f:
                        json.dump({"eval": {}, "predictions": prediction_rows}, f, ensure_ascii=False, indent=2)
                    print_rank_0(f"Saved final-test predictions to {prediction_file}", self.args.global_rank)


    def train_one_task(self, task, i_task, epochs):
        # 在单独某个任务上训练
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
        
        #### TRAIN ####
        train_dataloader = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        global_step = 0
        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                global_step += 1
                del batch['sources']
                batch = to_device(batch, device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
                    logging_steps = getattr(self.args, 'logging_steps', 10)
                    if global_step % logging_steps == 0:
                        print_rank_0(f"task={task} epoch={epoch+1} step={global_step} loss={loss.item():.6f}", self.args.global_rank)

                self.model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                self.model.step()

            # Validate on eval split after each epoch
            if not getattr(self.args, "disable_epoch_eval", False):
                print_rank_0(
                    f"***** Evaluating generation metrics, Epoch {epoch+1}/{epochs} on task {task} *****",
                    self.args.global_rank)
                eval_result, eval_predictions = self.task_generation_evaluation(
                    task,
                    eval_dataloader,
                    device,
                    max_ans_len=self._resolve_max_ans_len(i_task),
                    return_predictions=True,
                )
                print_rank_0(f"[task={task}] validation result: {eval_result}", self.args.global_rank)

                if self.args.global_rank == 0 and self.args.output_dir is not None:
                    safe_task_name = str(task).replace("/", "_").replace(":", "_")
                    pred_dir = os.path.join(self.args.output_dir, "predictions", f"eval-epoch{epoch+1}")
                    os.makedirs(pred_dir, exist_ok=True)
                    pred_file = os.path.join(pred_dir, f"{safe_task_name}.json")
                    with open(pred_file, "w", encoding="utf-8") as f:
                        json.dump({"metrics": eval_result, "predictions": eval_predictions}, f, ensure_ascii=False, indent=2)
                    print_rank_0(f"Saved eval predictions to {pred_file}", self.args.global_rank)
    
    
    def train_continual(self):
        for i_task, task in enumerate(self.train_task_list):
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))
            self.save_model(i_task)
        self.test_all_tasks_and_save_predictions()

    
    def save_model(self, round):
        if self.args.output_dir is not None:
            print_rank_0('saving model to ' + self.args.output_dir + "/" + str(round) + '...', self.args.global_rank)

        if self.args.global_rank == 0:
            save_hf_format(self.model, self.tokenizer, self.args, sub_folder=str(round))

        if self.args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(self.model,
                                  self.args.global_rank,
                                  self.args.output_dir,
                                  zero_stage=self.args.zero_stage,
                                  sub_folder=str(round))
        print_rank_0('Sucessful saving model after round {}'.format(round), self.args.global_rank)
        
