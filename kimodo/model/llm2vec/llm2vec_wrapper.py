# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM2Vec encoder wrapper for Kimodo text conditioning."""

import gc
import platform
import os
import numpy as np
import torch
from torch import nn
from .llm2vec import LLM2Vec

class LLM2VecEncoder(nn.Module):
    """LLM2Vec text embeddings."""

    def __init__(
        self,
        base_model_name_or_path: str,
        peft_model_name_or_path: str,
        dtype: str,
        llm_dim: int,
    ) -> None:
        super().__init__()
        self.torch_dtype = getattr(torch, dtype)
        self.llm_dim = llm_dim
        self.cpu_load = True  # default to loading on CPU until first use
        
        custom_path = r"path_to_your_Llama_text-encoders"
        if os.path.exists(custom_path):
            self.custom_dir = custom_path
        else:
            root_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir, os.pardir))
            self.custom_dir = os.path.abspath(os.path.join(root_path, "models", "KIMODO-Meta3_llm2vec_NF4"))
        
        print(f"[LLM2VecEncoder] Initializing model from {self.custom_dir}...")
        print(f"[LLM2VecEncoder] Initialized (Waiting for first use to load weights)...")
        self.model = None

    def unload(self):
        """Offload the model weights to System RAM (CPU) if currently on GPU."""
        if self.model is not None:
            if self.get_device().type == "cuda":
                print(f"[LLM2VecEncoder] Offloading 5.4GB model to System RAM...")
                self.model.model.to("cpu")
                gc.collect()
                if platform.system() == "Linux":
                    try:
                        import ctypes
                        ctypes.CDLL("libc.so.6").malloc_trim(0)
                    except Exception:
                        pass
                elif platform.system() == "Windows":
                    from kimodo.demo.memory_manager import release_system_memory
                    release_system_memory()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

    def reload(self):
        """Move from System RAM to VRAM."""
        if self.model is None:
            print(f"[LLM2VecEncoder] Model was None. Reloading from disk (15s delay)...")
            self.curr_device = self.get_device()
            self.model = LLM2Vec.from_pretrained(
                base_model_name_or_path=self.custom_dir,
                peft_model_name_or_path=None,
                torch_dtype=self.torch_dtype,
                device_map=self.curr_device
            )

        from kimodo.demo.memory_manager import manager
        # CPU load Model - cuda:0
        manager.ensure_vram_capacity(5400 * 1024 * 1024, device="cpu", exclude_name="text_encoder")

        
            
        gc.collect()
        
        if platform.system() == "Linux":
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
        elif platform.system() == "Windows":
            from kimodo.demo.memory_manager import release_system_memory
            release_system_memory()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        manager.log_memory_usage("Encoder Transfer Complete (RAM Reclaimed)")
        
        print(f"[LLM2VecEncoder] Model already on ({self.curr_device})")

    def get_device(self):
        device = 'cpu'
        if self.model is None or self.cpu_load:
            print(f"[LLM2VecEncoder] Model is currently on CPU. Using CPU for inference...")
            device = 'cpu'
        elif torch.backends.mps.is_available():
            print(f"[LLM2VecEncoder] Moving weights to GPU (mps)...")
            device = 'mps'
        elif torch.cuda.is_available():
            print(f"[LLM2VecEncoder] Moving weights to GPU (cuda:0)...")
            device = 'cuda:0'
        return torch.device(device)

    def delete(self):
        """Reclaim RAM without deleting from disk unless absolutely necessary."""
        self.unload()

    def __call__(self, text: list[str] | str):
        self.reload() # Auto-reload if called
        is_string = False
        if isinstance(text, str):
            text = [text]
            is_string = True

        results = []
        with torch.no_grad():
            for t in text:
                emb = self.model.encode([t])
                results.append(emb)

        encoded_text = np.concatenate(results, axis=0)

        assert len(encoded_text.shape)
        assert self.llm_dim == encoded_text.shape[-1]

        encoded_text = encoded_text[:, None]
        lengths = np.ones(len(encoded_text), dtype=int).tolist()

        if is_string:
            encoded_text = encoded_text[0]
            lengths = lengths[0]

        encoded_text = torch.tensor(encoded_text).to(self.get_device())
        return encoded_text, lengths