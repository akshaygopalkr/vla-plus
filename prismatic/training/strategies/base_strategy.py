"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, List

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast
from itertools import cycle
import torch.nn.functional as F

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.action_tokenizer import ActionTokenizer
import gc
import os

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
            self,
            vlm: PrismaticVLM,
            device_id: int,
            stage: str,
            epochs: int,
            max_steps: Optional[int],
            global_batch_size: int,
            per_device_batch_size: int,
            learning_rate: float,
            weight_decay: float,
            max_grad_norm: float,
            lr_scheduler_type: str,
            warmup_ratio: float,
            enable_gradient_checkpointing: bool = True,
            enable_mixed_precision_training: bool = True,
            reduce_in_full_precision: bool = False,
            mixed_precision_dtype: torch.dtype = torch.bfloat16,
            worker_init_fn: Optional[Callable[[int], None]] = None,
            **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
                self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
            self,
            run_dir: Path,
            global_step: int,
            epoch: int,
            train_loss: Optional[float] = None,
            only_trainable: bool = True,
    ) -> None:
        ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        ...

    @abstractmethod
    def clip_grad_norm(self) -> None:
        ...

    def run_training(
            self,
            dataset: Dataset,
            collator: PaddedCollatorForLanguageModeling,
            metrics: Metrics,
            stage: str = "finetune",
            batch_construction_strategy: str = "split-modality",
            seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
                total=(
                        (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                        if self.max_steps is None
                        else self.max_steps
                ),
                desc=status,
                leave=False,
                disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                            "cuda",
                            dtype=self.mixed_precision_dtype,
                            enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()

    # === VLA Training ===

    def run_vla_training(
            self,
            vla_dataset: IterableDataset,
            collator: PaddedCollatorForActionPrediction,
            action_tokenizer: ActionTokenizer,
            metrics: VLAMetrics,
            save_interval: int = 2500,
            save_full_model: bool = True,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        if self.max_steps is None:
            overwatch.info(f"We will train for {self.epochs} epochs and {self.epochs * len(dataloader)} steps!")
        else:
            overwatch.info(f"We will train for {self.max_steps} steps!")

        # === Train ===
        status = metrics.get_status()
        with tqdm(
                total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
                desc=status,
                leave=False,
                disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for batch in dataloader:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast(
                        "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = output.loss

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                loss.backward()

                # === Compute Action Token Accuracy & L1 Loss ===

                # To compute action token accuracy, we need to identify the locations of the action tokens
                # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                # insert `self.vlm.vision_backbone.num_patches` at index 1.
                #
                # Computing `action_prediction_accuracy` is then pretty straightforward:
                #   1) Extract "aligned" predictions & labels
                #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
                action_preds = output.logits[:, self.vlm.vision_backbone.num_patches: -1].argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                # Commit Metrics
                metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)

                # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                if overwatch.is_rank_zero():
                    datasets = set(batch["dataset_names"])
                    if len(datasets) > 1:
                        for ds in datasets:
                            ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                            action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()
                            continuous_actions_pred_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_preds[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            continuous_actions_gt_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_gt[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            action_l1_loss_ds = torch.nn.functional.l1_loss(
                                continuous_actions_pred_ds, continuous_actions_gt_ds
                            )
                            metrics.commit_for_dataset(
                                dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
                            )

                # === Gradient Step ===

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                        (metrics.global_step % save_interval) == 0
                ):
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    dist.barrier()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(status)

    # === Run VLA Co-Training ===

    def run_vla_cotraining(
            self,
            vla_dataset: IterableDataset,
            cotraining_datasets: List[IterableDataset],
            cotraining_dataset_names: List[str],
            collator: PaddedCollatorForActionPrediction,
            action_tokenizer: ActionTokenizer,
            metrics: VLAMetrics,
            save_interval: int = 2500,
            save_full_model: bool = True,
    ) -> None:

        def distribute_batches(batch_size):
            # Calculate the base batch size per device
            base_batch_size = batch_size // len(cotraining_datasets)
            # Calculate the remainder to distribute unevenly
            remainder = batch_size % len(cotraining_datasets)

            # Create the per-device batch sizes
            batch_sizes = [base_batch_size + (1 if i < remainder else 0) for i in range(len(cotraining_datasets))]

            return batch_sizes

        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert all(
            [isinstance(ds, IterableDataset) for ds in cotraining_datasets]), "Co-Training expects IterableDatasets!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        vla_dataloader = DataLoader(
            vla_dataset,
            batch_size=int(self.per_device_batch_size // 2),
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        cotraining_iterators = []
        for dataset, bs in zip(cotraining_datasets, distribute_batches(self.per_device_batch_size // 2)):
            cotraining_dataloader = DataLoader(
                dataset,
                batch_size=bs,
                sampler=None,
                collate_fn=collator,
                num_workers=0,
                worker_init_fn=self.worker_init_fn,
            )
            cotraining_iterators.append(iter(cotraining_dataloader))

        if self.max_steps is None:
            overwatch.info(f"We will train for {self.epochs} epochs and {self.epochs * len(vla_dataloader)} steps!")
        else:
            overwatch.info(f"We will train for {self.max_steps} steps!")

        def make_pixel_values(batch, i):

            if isinstance(batch["pixel_values"], dict):
                return {
                            "dino": batch["pixel_values"]['dino'][i],
                            "siglip": batch["pixel_values"]['siglip'][i]
                        }
            else:
                return batch["pixel_values"][i]

        def batch_list(batch):
            return [None] if not batch else [
                {
                    "input_ids": batch["input_ids"][i],
                    "pixel_values": make_pixel_values(batch, i),
                    "labels": batch["labels"][i],
                    "dataset_name": batch["dataset_names"][i]
                }
                for i in range(len(batch["input_ids"]))
            ]

        # === Train ===
        status = metrics.get_status()
        with tqdm(
                total=(self.epochs * len(vla_dataloader)) if self.max_steps is None else self.max_steps,
                desc=status,
                leave=False,
                disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for vla_batch in vla_dataloader:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!

                # Join the separate batches into a single one
                vla_batch_list = batch_list(vla_batch)

                co_training_batch_list = []
                co_training_batch_sizes = []
                for i in range(len(cotraining_iterators)):

                    # Reset iterator if we've reached the end
                    try:
                        cotraining_batch = next(cotraining_iterators[i])
                    except StopIteration:
                        cotraining_dataloader = DataLoader(
                            cotraining_datasets[i],
                            batch_size=distribute_batches(self.per_device_batch_size // 2)[i],
                            sampler=None,
                            collate_fn=collator,
                            num_workers=0,
                            worker_init_fn=self.worker_init_fn,
                        )
                        cotraining_iterators[i] = iter(cotraining_dataloader)  # Reset iterator
                        cotraining_batch = next(cotraining_iterators[i])

                    co_training_batch_list += batch_list(cotraining_batch)

                    if cotraining_batch:
                        co_training_batch_sizes.append(len(cotraining_batch["input_ids"]))
                    else:
                        overwatch.info(f"{cotraining_dataset_names[i]} is empty!", ctx_level=1)
                        co_training_batch_sizes.append(0)

                if all([b is None for b in co_training_batch_list]):
                    overwatch.info("Skipping empty batch")
                    continue
                else:
                    co_training_batch_list = [b for b in co_training_batch_list if b is not None]
                batch = collator(vla_batch_list + co_training_batch_list)

                # Shuffle the batch
                indices = torch.randperm(len(batch["input_ids"]))
                for key in batch:

                    if key == 'pixel_values':

                        if isinstance(batch[key], dict):
                            for sub_key in batch[key]:
                                batch[key][sub_key] = batch[key][sub_key][indices]
                        else:
                            batch[key] = batch[key][indices]
                    elif key == 'dataset_names':
                        batch[key] = [batch[key][i] for i in indices]
                    else:
                        batch[key] = batch[key][indices]

                # Get all the original VLA dataset indices for the batch before shuffling
                vla_indices = torch.where(indices < len(vla_batch_list))[0]
                cotraining_indice_dict = {}
                for i in range(len(cotraining_datasets)):

                    if co_training_batch_sizes[i] == 0:
                        continue
                    start = len(vla_batch_list) + sum(co_training_batch_sizes[:i])
                    end = len(vla_batch_list) + sum(co_training_batch_sizes[:i + 1])
                    cotraining_indice_dict[cotraining_dataset_names[i]] = \
                    torch.where(((indices >= start) & (indices < end)))[0]

                with torch.autocast(
                        "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):

                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = output.loss

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                loss.backward()

                # === Compute Action Token Accuracy & L1 Loss ===

                # To compute action token accuracy, we need to identify the locations of the action tokens
                # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                # insert `self.vlm.vision_backbone.num_patches` at index 1.
                #
                # Computing `action_prediction_accuracy` is then pretty straightforward:
                #   1) Extract "aligned" predictions & labels
                #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!

                # Only use the VLA indices for the action prediction
                action_preds = output.logits[vla_indices, self.vlm.vision_backbone.num_patches: -1].argmax(dim=2)
                action_gt = batch["labels"][vla_indices, 1:].to(action_preds.device)
                mask = action_gt > 29502
                action_preds = F.pad(action_preds, (action_gt.shape[-2] - action_preds.shape[-2],
                                                    action_gt.shape[-1] - action_preds.shape[-1]), "constant", 0)

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Commit Metrics
                # metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)
                metrics.commit(action_accuracy=action_accuracy, update_step_time=True)

                # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                if overwatch.is_rank_zero():
                    datasets = set(batch["dataset_names"])
                    if len(datasets) > 1:
                        for i, ds in enumerate(datasets):

                            indices = cotraining_indice_dict[
                                ds.decode()] if ds.decode() in cotraining_dataset_names else vla_indices
                            output_logits = output.logits[indices, self.vlm.vision_backbone.num_patches: -1]
                            labels = batch["labels"][indices, 1:].to(output_logits.device)
                            ds_loss = F.cross_entropy(output_logits.view(-1, output_logits.shape[2]), labels.view(-1))

                            if ds.decode() in cotraining_dataset_names:

                                action_preds = output.logits[indices, self.vlm.vision_backbone.num_patches: -1].argmax(
                                    dim=2)
                                action_gt = batch["labels"][indices, 1:].to(action_preds.device)
                                ds_mask = (action_gt > 2) & (action_gt != -100)
                                action_preds = F.pad(action_preds, (action_gt.shape[-2] - action_preds.shape[-2],
                                                                    action_gt.shape[-1] - action_preds.shape[-1]),
                                                     "constant", 0)
                                correct_preds = (action_preds == action_gt) & ds_mask
                                action_accuracy_ds = correct_preds.sum().float() / ds_mask.sum().float()
                                metrics.commit_for_dataset(
                                    dataset_name=ds.decode(), action_accuracy=action_accuracy_ds,
                                    loss=ds_loss
                                )

                            else:
                                action_accuracy_ds = action_accuracy
                                metrics.commit_for_dataset(
                                    dataset_name=ds.decode(), action_accuracy=action_accuracy_ds,
                                    loss=ds_loss
                                )

                # === Gradient Step ===

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                del output, batch
                gc.collect()

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                        (metrics.global_step % save_interval) == 0
                ):
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    dist.barrier()
                    gc.collect()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(status)
