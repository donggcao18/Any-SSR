import torch
from tqdm import tqdm

from model.base_model import CL_Base_Model
from utils.utils import print_rank_0, to_device


class MTL(CL_Base_Model):
    def train_continual(self):
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)

        use_fp16 = bool(getattr(self.args, "fp16", False))
        if use_fp16:
            self.model.half()

        train_dataloader = next(iter(self.train_task_list.values()))
        epochs = int(self.args.num_train_epochs[0])
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        global_step = 0

        for epoch in range(epochs):
            train_sampler = getattr(train_dataloader, "sampler", None)
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            print_rank_0(
                f"Beginning of MTL Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank,
            )
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                global_step += 1
                del batch["sources"]
                batch = to_device(batch, device)
                with torch.amp.autocast("cuda", enabled=use_fp16):
                    outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss

                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"MTL Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}",
                        refresh=False,
                    )
                    logging_steps = getattr(self.args, "logging_steps", 10)
                    if global_step % logging_steps == 0:
                        print_rank_0(
                            f"task=MTL epoch={epoch+1} step={global_step} loss={loss.item():.6f}",
                            self.args.global_rank,
                        )

                self.model.backward(loss)
                self.model.step()

            for eval_idx, (task, eval_dataloader) in enumerate(self.eval_task_list.items()):
                print_rank_0(
                    f"***** MTL validating on task {task}, Epoch {epoch+1}/{epochs} *****",
                    self.args.global_rank,
                )
                eval_result, eval_predictions = self.task_generation_evaluation(
                    task,
                    eval_dataloader,
                    device,
                    max_ans_len=self._resolve_max_ans_len(eval_idx),
                    return_predictions=True,
                )
                print_rank_0(f"[task={task}] MTL validation result: {eval_result}", self.args.global_rank)
                self._save_generation_predictions(
                    f"mtl-eval-epoch{epoch+1}",
                    eval_idx,
                    task,
                    eval_result,
                    eval_predictions,
                )

            self.save_model(f"epoch-{epoch+1}")
        self.test_all_tasks_and_save_predictions()
