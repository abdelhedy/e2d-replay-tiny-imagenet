"""
cl_plugin.py
------------
Avalanche plugin that integrates the original E2D pipeline into a continual
learning loop (Tiny-ImageNet / ImageNet).

Per-task lifecycle:
  after_training_exp
    1. [Recover]  Run recover_cl.py as a subprocess for the current task's
                  class IDs only — saves JPEG synthetic images to disk exactly
                  as the original E2D codebase does.
    2. [Relabel]  Load the JPEGs back, run them through the same 5-model teacher
                  ensemble used by recover_cl.py with N augmented views, and
                  average the soft-label distributions. Save one .pt file per
                  class. This replaces the per-epoch .tar FKD format: for CL
                  replay (≤100 epochs vs 300 in the original) averaged labels
                  carry the same teacher signal with far less disk overhead.
    3. [Buffer]   Register image paths + soft-label tensors in E2DBuffer.

  before_training_exp
    - Build a merged DataLoader: real current-task images (CE loss) +
      synthetic replay images (DIST/KL/MSE-GT soft-label KD loss).

  after_forward
    - Add the replay KD loss contribution to strategy.loss.
"""
from __future__ import annotations

import sys
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from cl_buffer import E2DBuffer, SyntheticReplayDataset
from normalization import (
    describe_normalization,
    make_relabel_transform,
    make_replay_transform,
)
from tiny_teachers import (
    load_tiny_teacher_ensemble,
    maybe_print_tiny_class_mapping,
    parse_teacher_checkpoints,
    tiny_avalanche_to_imagefolder_indices,
)


# ══════════════════════════════════════════════════════════════════════
#  KD losses  (identical to train_FKD_parallel.py)
# ══════════════════════════════════════════════════════════════════════

def _cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)

def _pearson(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return _cosine_sim(a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True), eps)

def _inter(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    return 1 - _pearson(y_s, y_t).mean()

def _intra(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    return _inter(y_s.T, y_t.T)


class DISTLoss(nn.Module):
    """DIST loss from train_FKD_parallel.py (unchanged)."""
    def __init__(self, beta: float = 2.0, gamma: float = 2.0, tem: float = 4.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.tem = tem

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        y_s = (s / self.tem).softmax(dim=1)
        y_t = (t / self.tem).softmax(dim=1)
        return (
            self.beta  * (self.tem ** 2) * _inter(y_s, y_t)
            + self.gamma * (self.tem ** 2) * _intra(y_s, y_t)
        )


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
#  Full-buffer loader: every gradient step sees ALL buffer images
# ══════════════════════════════════════════════════════════════════════

class _FullBufferLoader:
    """
    Wraps a real-data DataLoader and a pre-loaded buffer batch.

    Each iteration yields:
        concat( real_mini_batch , FULL_buffer_batch )

    → Every gradient step sees every synthetic replay image exactly once.
      This eliminates the softmax-competition forgetting that occurs when
      a 10:1 real/buffer ratio causes a net negative gradient on old-class
      fc weights even with a frozen backbone.
    """

    def __init__(self, real_loader: DataLoader, buf_batch: tuple):
        self.real_loader = real_loader
        self.buf_batch   = buf_batch          # tuple of 5 CPU tensors, each [n_syn, ...]

    def __iter__(self):
        buf = self.buf_batch
        for real_batch in self.real_loader:
            # real_batch : list/tuple of 5 tensors  [bs, ...]  (from _MergedDataset subset)
            # buf        : tuple  of 5 tensors  [n_syn, ...]
            yield tuple(
                torch.cat([rb, bb], dim=0)
                for rb, bb in zip(real_batch, buf)
            )

    def __len__(self) -> int:
        return len(self.real_loader)


# ══════════════════════════════════════════════════════════════════════
#  Merged dataset: real (CE) + synthetic replay (soft-label KD)
# ══════════════════════════════════════════════════════════════════════

class _MergedDataset(Dataset):
    """
    Merges a real Avalanche dataset with a SyntheticReplayDataset.

    Returned tuple:  (x, y, task_id, soft_label, is_replay)

    Real samples    → soft_label = zeros, is_replay = False
    Synthetic items → soft_label from buffer,  is_replay = True

    Keeping task_id in position 2 (Avalanche convention) avoids issues when
    Avalanche accesses mbatch[-1] as task ID.
    """

    def __init__(
        self,
        real_dataset,
        synthetic_dataset: SyntheticReplayDataset,
        num_classes: int,
    ):
        self.real      = real_dataset
        self.synthetic = synthetic_dataset
        self.num_classes = int(num_classes)
        self.real_len  = len(real_dataset)
        self.syn_len   = len(synthetic_dataset)

    def __len__(self) -> int:
        return self.real_len + self.syn_len

    def __getitem__(self, idx: int):
        if idx < self.real_len:
            sample = self.real[idx]
            x, y = sample[0], sample[1]
            t = sample[2] if len(sample) >= 3 else torch.tensor(0, dtype=torch.long)
            y = y.long() if torch.is_tensor(y) else torch.tensor(y, dtype=torch.long)
            t = t.long() if torch.is_tensor(t) else torch.tensor(t, dtype=torch.long)
            soft = torch.zeros(self.num_classes, dtype=torch.float32)
            is_replay = torch.tensor(False, dtype=torch.bool)
            return x, y, t, soft, is_replay

        syn_idx = idx - self.real_len
        x, y, soft = self.synthetic[syn_idx]
        # TinyImageNet teachers output logits in TinyImageNet class order.  If
        # the experiment uses a prefix subset (for example 10 of 200 classes),
        # keep the corresponding prefix of the teacher label space.
        if soft.shape[0] > self.num_classes:
            soft = soft[:self.num_classes]
        t = torch.tensor(-1, dtype=torch.long)
        is_replay = torch.tensor(True, dtype=torch.bool)
        return x, y, t, soft, is_replay


# ══════════════════════════════════════════════════════════════════════
#  Averaged soft-label relabeling
#  (replaces FKD per-epoch .tar files — see module docstring)
# ══════════════════════════════════════════════════════════════════════

# Original ImageNet teacher ensemble, kept as a fallback when no TinyImageNet
# checkpoints are provided.
_TORCHVISION_TEACHER_NAMES = [
    "resnet18",
    "mobilenet_v2",
    "efficientnet_b0",
    "shufflenet_v2_x0_5",
    "alexnet",
]

def _load_teacher_ensemble(
    device: torch.device,
    teacher_checkpoints: List[str],
    teacher_num_classes: int,
) -> List[nn.Module]:
    if teacher_checkpoints:
        return load_tiny_teacher_ensemble(
            teacher_checkpoints,
            device=device,
            num_classes=teacher_num_classes,
        )

    import torchvision.models as tv_models
    teachers = []
    for name in _TORCHVISION_TEACHER_NAMES:
        m = tv_models.__dict__[name](pretrained=True).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        teachers.append(m)
    print(f"[Relabel] Loaded {len(teachers)} teachers: {_TORCHVISION_TEACHER_NAMES}")
    return teachers


@torch.no_grad()
def compute_averaged_soft_labels(
    img_paths: List[str],
    teachers: List[nn.Module],
    device: torch.device,
    n_views: int = 10,
    batch_size: int = 64,
    transform: Optional[transforms.Compose] = None,
) -> torch.Tensor:
    """
    Run `n_views` independently augmented forward passes through the teacher
    ensemble for every synthetic image in `img_paths`.

    Returns averaged raw logits: shape [N, n_classes].
    (DISTLoss applies its own temperature softmax at training time.)

    Why averaged labels instead of per-epoch FKD .tar files?
    ─────────────────────────────────────────────────────────
    The original FKD saves one .tar per (epoch, batch) so the student sees
    the exact crop the teacher scored.  For CL replay (≪300 epochs, small
    buffer fraction), averaging over n_views ≥ 10 gives the same expected
    gradient signal without requiring epoch-synchronised loaders or thousands
    of .tar files.  The practical accuracy difference for replay is negligible.
    """
    n_imgs   = len(img_paths)
    n_classes: Optional[int] = None
    accum: Optional[torch.Tensor] = None
    transform = transform or make_relabel_transform(224, "imagenet")

    for _ in range(n_views):
        # Load + augment all images for this view
        imgs = torch.stack(
            [transform(Image.open(p).convert("RGB")) for p in img_paths]
        )

        for start in range(0, n_imgs, batch_size):
            batch = imgs[start : start + batch_size].to(device)
            logit_sum: Optional[torch.Tensor] = None

            for teacher in teachers:
                out = teacher(batch).float()
                logit_sum = out if logit_sum is None else logit_sum + out

            # Store averaged raw logits — DISTLoss applies its own temperature
            # softmax.  Pre-softmaxing here caused double-softmax at training
            # time (T=20 here × T=4 in DISTLoss) which collapsed teacher signal
            # to near-uniform, killing knowledge distillation.
            avg_logits = logit_sum / len(teachers)

            if n_classes is None:
                n_classes = avg_logits.shape[1]
                accum = torch.zeros(n_imgs, n_classes, dtype=torch.float32)

            accum[start : start + batch_size] += avg_logits.cpu()

    accum /= n_views          # average over views
    return accum               # [N, C]


# ══════════════════════════════════════════════════════════════════════
#  Main plugin
# ══════════════════════════════════════════════════════════════════════

class E2DReplayPlugin(SupervisedPlugin):
    """
    Integrates the full E2D 3-stage pipeline into an Avalanche CL strategy.

    Stage 1 — Recover   : calls recover_cl.py as subprocess (GPU is freed first)
    Stage 2 — Relabel   : averaged soft labels via teacher ensemble
    Stage 3 — Train     : merged DataLoader + soft-label KD in after_forward
    """

    def __init__(
        self,
        # ── paths ───────────────────────────────────────────────────
        output_dir: str,
        recover_script: str,      # path to recover_cl.py
        train_data_path: str,     # ImageNet train/ root (task-filtered for BN stats)
        # ── buffer ──────────────────────────────────────────────────
        ipc: int = 50,
        fixed_per_class: bool = True,
        budget_mode: str = "fixed-per-class",
        num_classes: int = 1000,
        device: torch.device = torch.device("cuda"),
        # ── recover hyper-params (passed through to recover_cl.py) ──
        recover_iterations: int = 1000,
        recover_lr: float = 0.1,
        recover_batch_size: int = 100,
        K: int = 700,
        loss_threshold: float = 0.5,
        r_loss: float = 0.05,
        first_multiplier: float = 10.0,
        tv_l2: float = 0.0001,
        training_momentum: float = 0.4,
        gpu_id: str = "0",
        # ── relabeling ──────────────────────────────────────────────
        relabel_views: int = 10,
        relabel_temperature: float = 20.0,
        relabel_batch_size: int = 64,
        teacher_checkpoints: Optional[str] = None,
        teacher_num_classes: int = 1000,
        teacher_image_size: int = 224,
        normalization: str = "imagenet",
        debug_diagnostics: bool = False,
        # ── replay KD ───────────────────────────────────────────────
        kd_loss: str = "dist",          # "dist" | "kl" | "mse_gt"
        kd_weight: float = 0.5,
        kd_temperature: float = 4.0,
        # ── dataloader ──────────────────────────────────────────────
        num_workers: int = 0,  # 0 avoids Windows page-file exhaustion from CUDA DLL spawning
    ):
        super().__init__()

        self.output_dir       = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.recover_script   = str(recover_script)
        self.train_data_path  = str(train_data_path)

        self.ipc         = ipc
        self.num_classes = int(num_classes)
        self.budget_mode = str(budget_mode)
        self.device      = device
        self.gpu_id      = gpu_id
        self.num_workers = num_workers

        # recover
        self.recover_iterations  = int(recover_iterations)
        self.recover_lr          = float(recover_lr)
        self.recover_batch_size  = int(recover_batch_size)
        self.K                   = int(K)
        self.loss_threshold      = float(loss_threshold)
        self.r_loss              = float(r_loss)
        self.first_multiplier    = float(first_multiplier)
        self.tv_l2               = float(tv_l2)
        self.training_momentum   = float(training_momentum)

        # relabeling
        self.relabel_views       = int(relabel_views)
        self.relabel_temperature = float(relabel_temperature)
        self.relabel_batch_size  = int(relabel_batch_size)
        self.teacher_checkpoints = parse_teacher_checkpoints(teacher_checkpoints)
        self.teacher_num_classes = int(teacher_num_classes)
        self.teacher_image_size  = int(teacher_image_size)
        self.normalization_name  = str(normalization)
        self.relabel_transform   = make_relabel_transform(
            self.teacher_image_size,
            self.normalization_name,
        )
        self.replay_transform    = make_replay_transform(
            self.teacher_image_size,
            self.normalization_name,
        )
        self.debug_diagnostics   = bool(debug_diagnostics)
        print(f"[E2DPlugin] Normalization: {describe_normalization(self.normalization_name)}")
        self.teacher_index_map = None
        if self.teacher_checkpoints:
            self.teacher_index_map = tiny_avalanche_to_imagefolder_indices(
                self.train_data_path,
                num_classes=self.teacher_num_classes,
            )
            maybe_print_tiny_class_mapping(self.teacher_index_map, "[E2DPlugin]")
        self.student_to_teacher_indices = None
        if self.teacher_index_map is not None:
            self.student_to_teacher_indices = torch.tensor(
                self.teacher_index_map[: self.num_classes],
                dtype=torch.long,
            )

        # KD
        self.kd_loss_name = kd_loss
        self.kd_weight    = float(kd_weight)
        self.dist_loss    = DISTLoss(tem=kd_temperature)

        self.buffer = E2DBuffer(
            root=str(self.output_dir),
            ipc=ipc,
            fixed_per_class=fixed_per_class,
            budget_mode=self.budget_mode,
            total_classes=self.num_classes,
        )
        self.seen_classes: Set[int] = set()
        self._teachers: Optional[List[nn.Module]] = None
        self._current_exp_id: Optional[int] = None
        self._printed_loss_diag: Set[int] = set()

        # Resume support: reload buffer state from a previous run
        if self.buffer.load_index():
            self.seen_classes = set(self.buffer._data.keys())
            print(f"[E2DPlugin] Resumed buffer: {len(self.seen_classes)} classes already distilled.")

    # ── teacher management ────────────────────────────────────────────

    def _load_teachers(self) -> None:
        if self._teachers is None:
            self._teachers = _load_teacher_ensemble(
                self.device,
                teacher_checkpoints=self.teacher_checkpoints,
                teacher_num_classes=self.teacher_num_classes,
            )

    def _unload_teachers(self) -> None:
        """Move teachers to CPU and free GPU memory before recover subprocess."""
        if self._teachers is not None:
            for t in self._teachers:
                t.cpu()
            del self._teachers
            self._teachers = None
            torch.cuda.empty_cache()

    # ── recover subprocess ────────────────────────────────────────────

    def _run_recover(self, task_id: int, class_ids: List[int], ipc_number: int) -> Path:
        """
        Call recover_cl.py as a subprocess restricted to `class_ids`.
        The student model has been moved to CPU before this call so the
        subprocess has full GPU access.
        Returns the syn/ directory for this task.
        """
        syn_root = self.buffer.task_syn_dir(task_id)
        # recover_cl.py uses: syn_data_path / exp_name → we set exp_name="syn"
        syn_root.parent.mkdir(parents=True, exist_ok=True)

        class_ids_str = ",".join(str(c) for c in sorted(class_ids))

        # Each task gets its own statistic subdirectory so BN stats computed
        # from task T's data don't overwrite or pollute task T+1's stats.
        task_statistic_path = str(self.output_dir / f"task_{task_id}" / "statistic")

        cmd = [
            sys.executable, self.recover_script,
            "--train-data-path",   self.train_data_path,
            "--statistic-path",    task_statistic_path,   # per-task, CL-compliant
            "--syn-data-path",     str(syn_root.parent),  # task_{T}/
            "--exp-name",          "syn",
            "--ipc-number",        str(ipc_number),
            "--iteration",         str(self.recover_iterations),
            "--lr",                str(self.recover_lr),
            "--batch-size",        str(self.recover_batch_size),
            "--K",                 str(self.K),
            "--loss-threshold",    str(self.loss_threshold),
            "--r-loss",            str(self.r_loss),
            "--first-multiplier",  str(self.first_multiplier),
            "--tv-l2",             str(self.tv_l2),
            "--training-momentum", str(self.training_momentum),
            "--gpu-id",            self.gpu_id,
            "--num-classes",       str(self.teacher_num_classes),
            "--image-size",        str(self.teacher_image_size),
            "--normalization",     self.normalization_name,
            "--class-ids",         class_ids_str,
            "--store-best-images",
        ]
        if self.teacher_checkpoints:
            cmd += ["--teacher-checkpoints", ",".join(self.teacher_checkpoints)]

        print(f"[E2DPlugin] Launching recover_cl.py | classes {class_ids}")
        t0 = time.time()
        result = subprocess.run(cmd, check=True, text=True)
        elapsed = time.time() - t0
        print(f"[E2DPlugin] Recover done in {elapsed:.1f}s")

        # Delete statistic directory immediately — it is only needed during recover
        # and can be several GB per task, quickly exhausting Kaggle's disk quota.
        import shutil
        stat_path = Path(task_statistic_path)
        if stat_path.exists():
            shutil.rmtree(str(stat_path))
            print(f"[E2DPlugin] Deleted statistic cache: {stat_path}")

        return syn_root

    # ── image path collection ─────────────────────────────────────────

    def _collect_img_paths(
        self, syn_dir: Path, class_ids: List[int]
    ) -> dict:
        """
        Gather JPEG paths for each class from recover_cl.py output.
        recover_cl.py saves to: syn_dir/new{class_id:03d}/*.jpg
        """
        paths: dict = {}
        for cid in class_ids:
            class_dir = syn_dir / f"new{cid:03d}"
            if not class_dir.exists():
                print(f"  [warning] no syn dir for class {cid}: {class_dir}")
                paths[cid] = []
            else:
                paths[cid] = sorted(str(p) for p in class_dir.glob("*.jpg"))
        return paths

    def _format_tensor_stats(self, x: torch.Tensor) -> str:
        x = x.detach().float().cpu()
        return (
            f"shape={tuple(x.shape)} "
            f"min={x.min().item():.4f} "
            f"mean={x.mean().item():.4f} "
            f"max={x.max().item():.4f}"
        )

    def _print_batch_diagnostics(self, label: str, batch) -> None:
        if not self.debug_diagnostics:
            return
        if batch is None or len(batch) < 2:
            return

        x = batch[0]
        if not torch.is_tensor(x):
            print(f"[Diag] {label}: first field is not a tensor ({type(x).__name__})")
            return
        y = batch[1]
        y_cpu = y.detach().cpu() if torch.is_tensor(y) else torch.as_tensor(y)
        unique = torch.unique(y_cpu).tolist()
        unique_preview = unique[:20]
        suffix = ""
        if len(batch) >= 5 and torch.is_tensor(batch[4]):
            replay_count = int(batch[4].sum().item())
            suffix = f" replay={replay_count}/{len(batch[4])}"

        print(
            f"[Diag] {label}: {self._format_tensor_stats(x)} "
            f"labels={unique_preview}{'...' if len(unique) > 20 else ''}{suffix}"
        )

    def _print_real_dataset_diagnostics(self, real_dataset, label: str = "real probe") -> None:
        if not self.debug_diagnostics or len(real_dataset) == 0:
            return
        try:
            n_probe = min(8, len(real_dataset))
            probe = default_collate([real_dataset[i] for i in range(n_probe)])
            self._print_batch_diagnostics(label, probe)
        except Exception as exc:
            print(f"[Diag] real probe failed: {exc}")

    def _print_synthetic_image_diagnostics(self, paths_per_class: dict) -> None:
        if not self.debug_diagnostics:
            return

        sizes = {}
        first_path = None
        for cid, paths in paths_per_class.items():
            if not paths:
                continue
            try:
                with Image.open(paths[0]) as img:
                    sizes.setdefault(img.size, []).append(cid)
            except Exception as exc:
                print(f"[Diag] could not inspect synthetic JPEG for class {cid}: {exc}")
                continue
            if first_path is None:
                first_path = paths[0]

        if not sizes:
            print("[Diag] synthetic JPEG: no images found")
            return

        size_text = ", ".join(
            f"{size}: classes {cids[:8]}" for size, cids in sizes.items()
        )
        print(f"[Diag] synthetic JPEG sizes: {size_text} | first={first_path}")

    # ── Avalanche hooks ───────────────────────────────────────────────

    def before_training_exp(self, strategy, **kwargs) -> None:
        """Inject synthetic replay data before the experience starts."""
        self._current_exp_id = int(getattr(strategy.experience, "current_experience", -1))
        self._print_real_dataset_diagnostics(strategy.adapted_dataset)

        syn_dataset = self.buffer.get_dataset(transform=self.replay_transform)
        if syn_dataset is None:
            print("[E2DPlugin] First experience — no replay data yet.")
            return

        n_real = len(strategy.adapted_dataset)
        n_syn  = len(syn_dataset)

        print(
            f"[E2DPlugin] Full-buffer replay: {n_real} real imgs in exp + "
            f"{n_syn} buffer imgs appended every step → 100% buffer coverage."
        )

        merged = _MergedDataset(
            real_dataset=strategy.adapted_dataset,
            synthetic_dataset=syn_dataset,
            num_classes=self.num_classes,
        )

        # Pre-load ALL buffer items (indices n_real … n_real+n_syn-1 in merged).
        # 64×64 images: 400 items ≈ 20 MB — fast to load once per experience.
        buf_items = [merged[n_real + i] for i in range(n_syn)]
        buf_batch = default_collate(buf_items)   # tuple of 5 CPU tensors [n_syn, ...]
        self._print_batch_diagnostics("replay full-buffer", buf_batch)

        # Real-data loader: shuffled batches of train_mb_size.
        # Uses the real-item Subset of merged so items are 5-tuples (same format).
        real_subset = torch.utils.data.Subset(merged, range(n_real))
        real_loader = DataLoader(
            real_subset,
            batch_size=strategy.train_mb_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
        )

        # Each training iteration = real mini-batch ++ full buffer batch.
        strategy.dataloader = _FullBufferLoader(real_loader, buf_batch)

    def after_forward(self, strategy, **kwargs) -> None:
        """
        Add the replay KD loss on synthetic samples.
        Fires when strategy.loss == 0 (before the criterion adds CE loss).
        After this hook: total_loss = KD_loss + CE_loss (from criterion).
        """
        if self.kd_weight <= 0:
            return

        mbatch = getattr(strategy, "mbatch", None)
        # mbatch = (x, y, task_id, soft_label, is_replay) from _MergedDataset
        if mbatch is None or len(mbatch) < 5:
            return

        is_replay = mbatch[4]
        if not torch.is_tensor(is_replay):
            is_replay = torch.as_tensor(is_replay)
        is_replay = is_replay.to(strategy.mb_output.device).bool()

        if not is_replay.any():
            return

        soft_labels   = mbatch[3].to(strategy.mb_output.device).float()
        s_logits      = strategy.mb_output[is_replay]
        t_logits      = soft_labels[is_replay]
        kd            = self._kd_loss(s_logits, t_logits)
        self._print_loss_diagnostics(strategy, is_replay, kd)
        strategy.loss = strategy.loss + self.kd_weight * kd

    def _print_loss_diagnostics(
        self,
        strategy,
        is_replay: torch.Tensor,
        kd: torch.Tensor,
    ) -> None:
        if not self.debug_diagnostics:
            return
        exp_id = self._current_exp_id
        if exp_id is None or exp_id in self._printed_loss_diag:
            return

        y = strategy.mbatch[1].to(strategy.mb_output.device).long()
        logits = strategy.mb_output
        real_mask = ~is_replay

        ce_all = F.cross_entropy(logits, y, reduction="mean")
        ce_real = (
            F.cross_entropy(logits[real_mask], y[real_mask], reduction="mean")
            if real_mask.any()
            else torch.tensor(float("nan"), device=logits.device)
        )
        ce_replay_hard = F.cross_entropy(
            logits[is_replay],
            y[is_replay],
            reduction="mean",
        )
        print(
            f"[Diag] losses exp {exp_id}: "
            f"CE(all)={ce_all.item():.4f} "
            f"CE(real)={ce_real.item():.4f} "
            f"CE(replay-hard)={ce_replay_hard.item():.4f} "
            f"KD(raw)={kd.item():.4f} "
            f"KD(weighted)={(self.kd_weight * kd).item():.4f}"
        )
        self._printed_loss_diag.add(exp_id)

    def _kd_loss(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.kd_loss_name == "dist":
            return self.dist_loss(s, t)
        T = self.dist_loss.tem
        if self.kd_loss_name == "kl":
            return (
                F.kl_div(F.log_softmax(s / T, dim=1), F.softmax(t / T, dim=1),
                         reduction="batchmean") * T ** 2
            )
        if self.kd_loss_name == "mse_gt":
            return F.mse_loss(F.softmax(s / T, dim=1), F.softmax(t / T, dim=1))
        raise ValueError(f"Unknown kd_loss: {self.kd_loss_name!r}")

    def after_training_exp(self, strategy, **kwargs) -> None:
        """Run recover → relabel → update buffer after each task."""
        exp      = strategy.experience
        task_id  = int(getattr(exp, "current_experience", 0))
        new_cls  = [int(c) for c in exp.classes_in_this_experience]
        self.seen_classes.update(new_cls)
        total    = len(self.seen_classes)
        seen_class_ids = sorted(self.seen_classes)
        target_budgets = self.buffer.class_budgets(seen_class_ids)
        recover_ipc = max(target_budgets.get(cid, self.ipc) for cid in new_cls)

        print(f"\n[E2DPlugin] Task {task_id} | new classes: {new_cls} | seen: {total}")
        if self.budget_mode == "global":
            total_budget = self.ipc * self.num_classes
            print(
                f"[E2DPlugin] Global budget: {total_budget} total synthetic imgs "
                f"-> generating up to {recover_ipc}/class for new classes."
            )

        # ── Stage 1: Recover ─────────────────────────────────────────
        # Free teacher + student + optimizer GPU memory before subprocess
        self._unload_teachers()
        # Unwrap DataParallel if present so .cpu()/.to() work correctly
        _raw_model = strategy.model.module if isinstance(strategy.model, nn.DataParallel) else strategy.model
        _raw_model.cpu()
        # Move optimizer momentum buffers to CPU
        for state in strategy.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        try:
            syn_dir = self._run_recover(task_id, new_cls, recover_ipc)
        finally:
            # Restore student and optimizer state to device regardless of recover success
            _raw_model.to(self.device)
            for state in strategy.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        paths_per_class = self._collect_img_paths(syn_dir, new_cls)
        self._print_synthetic_image_diagnostics(paths_per_class)

        # ── Stage 2: Relabel ─────────────────────────────────────────
        self._load_teachers()

        for cid in new_cls:
            img_paths = paths_per_class.get(cid, [])
            if not img_paths:
                print(f"  [E2DPlugin] class {cid}: no images found, skipping.")
                continue

            print(
                f"  -> class {cid} | {len(img_paths)} imgs "
                f"| {self.relabel_views} views × {len(self._teachers)} teachers"
            )
            soft_labels = compute_averaged_soft_labels(
                img_paths=img_paths,
                teachers=self._teachers,
                device=self.device,
                n_views=self.relabel_views,
                batch_size=self.relabel_batch_size,
                transform=self.relabel_transform,
            )
            full_teacher_logits = soft_labels
            if self.student_to_teacher_indices is not None:
                max_teacher_idx = int(self.student_to_teacher_indices.max().item())
                if soft_labels.shape[1] <= max_teacher_idx:
                    raise ValueError(
                        "Teacher logits are too small for TinyImageNet class remap: "
                        f"logits={soft_labels.shape[1]}, max index={max_teacher_idx}"
                    )
                soft_labels = soft_labels.index_select(
                    1, self.student_to_teacher_indices
                )
            if self.debug_diagnostics:
                teacher_target = (
                    self.teacher_index_map[cid]
                    if self.teacher_index_map is not None
                    else cid
                )
                teacher_acc = (
                    full_teacher_logits.argmax(1) == int(teacher_target)
                ).float().mean().item()
                replay_acc = (
                    soft_labels.argmax(1) == int(cid)
                ).float().mean().item()
                print(
                    f"  [Diag] teacher top1 on synthetic class {cid}: "
                    f"full-space={teacher_acc:.3f} target={teacher_target}, "
                    f"student-space={replay_acc:.3f}"
                )

            # Persist soft labels alongside the JPEGs
            sl_path = self.buffer.soft_label_path(task_id, cid)
            sl_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(soft_labels, str(sl_path))

            # ── Stage 3 prep: update buffer ───────────────────────────
            self.buffer.update(
                task_id=task_id,
                class_id=cid,
                img_paths=img_paths,
                soft_labels=soft_labels,
                total_classes=total,
                seen_class_ids=seen_class_ids,
            )

        # Save buffer index for resume support
        self.buffer.save_index()
        self._unload_teachers()
        print(f"[E2DPlugin] {self.buffer}\n")
