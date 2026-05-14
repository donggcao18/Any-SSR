import torch
import torch.utils.data
from tqdm.auto import tqdm
from model.base_model import CL_Base_Model
from utils.utils import print_rank_0



class EWC(CL_Base_Model):
    def __init__(self,model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args, lambda_ewc=400):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        self.device="cuda"
        self.use_fp16 = bool(getattr(self.args, "fp16", False))
        if self.use_fp16:
            self.model.half()
        self.lambda_ewc = lambda_ewc
        self.trainable_param_names = [
            n for n, p in self.model.named_parameters() if p.requires_grad
        ]
        self._previous_params = {}
        self.fisher = {}
        self._hook_handles = []
        self.task_num = 0
        self.train_length = 1
        self._backward_loss_scale = 1.0

        self._update_previous_params()
        self.init_fisher()


    
    def init_fisher(self):
        for n, p in self.model.named_parameters():
            if n in self.trainable_param_names:
                self.fisher[n] = torch.zeros_like(
                    p.detach(),
                    device="cpu",
                    dtype=torch.float32,
                )
            
    #计算每个参数的Fisher信息矩阵的值：每个样本输入模型，每个参数计算梯度的平方和，除以总的样本数量
    def _update_fisher(self, name, grad):
        if name not in self.fisher:
            return
        with torch.no_grad():
            grad = torch.nan_to_num(grad.detach(), nan=0.0)
            if self._backward_loss_scale != 1.0:
                grad = grad / self._backward_loss_scale
            grad = grad.to(device="cpu", dtype=torch.float32)
            self.fisher[name].add_(grad.square(), alpha=1.0 / max(1, self.train_length))

    #正则化，除以训练集长度
    def _regular_fisher(self):
        for n in self.fisher:
            self.fisher[n].div_(self.train_length)

    
    def _update_previous_params(self):
        for n, p in self.model.named_parameters():
            if n in self.trainable_param_names:
                self._previous_params[n] = p.detach().cpu().clone() # Previous task parameters

    #计算惩罚loss
    def penalty(self):
        restrict_loss = torch.zeros((), device=self.device)
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.fisher:
                    previous = self._previous_params[n].to(
                        device=p.device,
                        dtype=p.dtype,
                        non_blocking=True,
                    )
                    fisher = self.fisher[n].to(
                        device=p.device,
                        dtype=p.dtype,
                        non_blocking=True,
                    )
                    restrict_loss += (fisher * (p.detach() - previous).square()).sum()
        return restrict_loss

    def _get_loss_scale(self):
        optimizer = getattr(self.model, "optimizer", None)
        loss_scaler = getattr(optimizer, "loss_scaler", None)
        for obj in (loss_scaler, optimizer):
            for attr in ("cur_scale", "loss_scale"):
                value = getattr(obj, attr, None)
                if value is not None:
                    try:
                        return float(value)
                    except TypeError:
                        return float(value.item())
        return 1.0
    
    def train_step(self,
                    batch):

        # batch = {k: batch[k].to(self.device) for k in batch}
        lm_labels = batch["labels"]
        # lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        # inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])
        # inputs_embeds = self.model.model.embed_tokens(batch["input_ids"])  #向量，【batch * embedding_size】

        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            outputs = self.model(
                input_ids=batch['input_ids'],
                labels=lm_labels,
                attention_mask=batch['attention_mask'],
                use_cache=False,
            )
        
        loss = outputs[0]

        return loss
    
    def save_grad(self, name, param):
        def hook(grad):
            clean_grad = torch.nan_to_num(grad, nan=0.0)
            adjusted_grad = clean_grad

            if self.task_num != 0 and name in self.fisher:
                previous = self._previous_params[name].to(
                    device=grad.device,
                    dtype=grad.dtype,
                    non_blocking=True,
                )
                fisher = self.fisher[name].to(
                    device=grad.device,
                    dtype=grad.dtype,
                    non_blocking=True,
                )
                ewc_grad = param.detach().to(dtype=grad.dtype) - previous
                ewc_grad.mul_(fisher)
                if self._backward_loss_scale != 1.0:
                    ewc_grad.mul_(self._backward_loss_scale)
                adjusted_grad.add_(ewc_grad, alpha=self.lambda_ewc)

            self._update_fisher(name, clean_grad)
            return adjusted_grad
        return hook

    def retain_grad(self):
        for n,p in self.model.named_parameters():
            if n in self.fisher.keys():
                self._hook_handles.append(p.register_hook(self.save_grad(n, p)))
    
    
    def train_one_task(self,
                       task,
                       i_task,
                       epochs=40):

        print_rank_0(f'task = {task}', self.args.global_rank)

        dataloader_train = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]
        self.train_length = len(dataloader_train)
        total_steps = epochs * len(dataloader_train)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        global_step = 0

        for epoch in range(epochs):
            print_rank_0(f'Epoch {epoch+1}/{epochs}', self.args.global_rank)
            self.model.train()

            for step, batch in enumerate(dataloader_train):
                global_step += 1
                del batch['sources']
                batch.pop('indices', None)
                batch = {k:batch[k].to('cuda') for k in batch}
                loss = self.train_step(batch)
                
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
                    if global_step % self.args.logging_steps == 0:
                        print_rank_0(f"task={task} epoch={epoch+1} step={global_step} loss={loss.item():.6f}", self.args.global_rank)
                self._backward_loss_scale = self._get_loss_scale()
                self.model.backward(loss)
                self.model.step()

            # Validate on eval split after each epoch.
            print_rank_0(
                f"***** Evaluating generation metrics, Epoch {epoch+1}/{epochs} on task {task} *****",
                self.args.global_rank)
            eval_result, eval_predictions = self.task_generation_evaluation(
                task,
                eval_dataloader,
                self.device,
                max_ans_len=self._resolve_max_ans_len(i_task),
                return_predictions=True,
            )
            print_rank_0(f"[task={task}] validation result: {eval_result}", self.args.global_rank)

            self._save_generation_predictions(f"eval-epoch{epoch+1}", i_task, task, eval_result, eval_predictions)


        for seen_idx, (test_task, test_dataset) in enumerate(list(self.test_task_list.items())[:i_task+1]):
            print_rank_0(
                f"***** Testing on current task {test_task} after training {task} on all epochs *****",
                self.args.global_rank)
            test_result, test_predictions = self.task_generation_evaluation(
                test_task,
                test_dataset,
                self.device,
                max_ans_len=self._resolve_max_ans_len(seen_idx),
                return_predictions=True,
            )
            print_rank_0(f"[task={test_task}] post-train test result: {test_result}", self.args.global_rank)

            self._save_generation_predictions("test-after-task", i_task, test_task, test_result, test_predictions)


    
    # Train model continually
    def train_continual(self):
        #在训练之前确定梯度
        self.retain_grad()

        for i_task, task in enumerate(self.train_task_list):
            self.task_num=i_task
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))
            # self._regular_fisher()
            
            self._update_previous_params()
            self.save_model(i_task)
        # self.test_all_tasks_and_save_predictions()
            
            
