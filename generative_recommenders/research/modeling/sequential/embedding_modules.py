# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import abc

import torch
from torch.utils.checkpoint import checkpoint

from generative_recommenders.research.modeling.initialization import truncated_normal


class EmbeddingModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class LocalSIDEmbeddingModule(EmbeddingModule):

    def __init__(self, 
                 emb_tbl_size: int,
                 num_items: int,
                 item_embedding_dim: int,
                 codebook: torch.Tensor,
                 lookup: torch.Tensor,
                 emb_method: str,
                 emb_comb_method: str,
                 scale_SID_emb: bool,
                 used_uniq_layer: bool,
                 use_num_codebook_layers: int
                 ) -> None:
        """ Uses semantic IDs to build embeddings for posts.
        
        Args:
            emb_tbl_size (int): size of the SID embedding table, e.g. 2000003
            num_items (int): number of unique posts (just like above)
            item_embedding_dim (int): dimension of the item embedding vec
            codebook (torch.Tensor): num_layers * num_codes * code_dimensions. This wouldn't be useful if posts are already in the lookup table, 
                but this will be used if #TODO: we want to skip the lookup table or if there are new posts and we need to calculate SIDs from scratch.
            lookup (torch.Tensor): num_items * num_layers.
            emb_method (str): "Ngram" or "prefixN" or "prefixN-indEmb". Let's say we have K codes per layer
                - "RQsum-indEmb": use L * C size embedding table, and then sum up the L embeddings extracted, also plus the indivisual embedding
                - "Ngram": K^2 * c1 + K^1 * c2 + K^0 * c3 for a SID of (c1, c2, c3)
                - "prefixN": obtain c1, K^1 * c1 + k^0 * c2, K^2 * c1 + K^1 * c2 + K^0 * c3 for a given SID of (c1, c2, c3), 
                    And then sum pooling all the corresponding embeddings.
                - "prefixN-indEmb": we are keeping both the prefixN embeddings from the SID embedding table, and the 
                    individual embeddings from the original item embedding table. (#TODO: check if memory is ok))
            emb_comb_method (str): how to combine the semantic embeddings and individual embeddings, 
                choose from: 
                - ["sum"]: earliest and default one, simply sum them together 
                - "FC":
                - "MLP": 
                - "scalarGate":
                - "vectorGate":
                - "scalarWeight":
            scale_SID_emb (bool): if True, then scale the SID embeddings by sqrt(num embs summed) 
            used_uniq_layer (bool): means there is an additional layer in the end (TIGER-style), so the codebook num_layers
                and the lookup table SID size differs by 1. won't be important if not using prefixN-indEmb.
            use_num_codebook_layers (int): if > 0, then only use the first N (or all) layers of the codebook, otherwise use all layers.
        
        """
        super().__init__()

        ## ========== embedding table and method init
        self._scale_SID_emb: bool = scale_SID_emb
        self._emb_method: str = emb_method  # how to get SID embeddings
        
        self._num_items = num_items
        self._item_embedding_dim: int = item_embedding_dim

        ## ========== SID codebook and lookup setup
        self.register_buffer("_SID_codebook", codebook) # in case in the future we want to trade off computations for memory, we can calc SID on the fly
        self._num_layers = codebook.shape[0] 
        if used_uniq_layer and (not emb_method in ["prefixN-indEmb", "RQsum-indEmb"]):  
            # TIGER-style codebook, the last layer does not have a code in the codebook, but the lookup table have that arbitrary number
            self._num_layers += 1
        if use_num_codebook_layers > 0:
            self._num_layers = min(self._num_layers, use_num_codebook_layers)

        self._num_codes = codebook.shape[1]
        print(f"** initialized SID emedding module: \n"
              f" - Embedding method: {self._emb_method};\n"
              f" - SID codebook layers: {self._num_layers}, codebook dimension per layer: {self._num_codes}\n")
        
        if lookup is not None:
            self.register_buffer("_lookup", lookup)
        else:
            self._lookup = None

        ## ========== embedding table, embedding combination method init
        if emb_method == "RQsum-indEmb":
            self._emb_tbl_size = self._num_layers * self._num_codes
            print(f"**!! Using RQsum-indEmb method, forced to set emb_tbl_size to num layers * num codes: {self._emb_tbl_size}")
        else:
            self._emb_tbl_size: int = emb_tbl_size

        self._item_emb_SID: torch.nn.Embedding = torch.nn.Embedding(self._emb_tbl_size + 1, item_embedding_dim, padding_idx=0)

        if emb_method in ["prefixN-indEmb", "RQsum-indEmb"]:
            # keeping the individual item embeddings as well
            self._item_emb_individual: torch.nn.Embedding = torch.nn.Embedding(num_items + 1, item_embedding_dim, padding_idx=0)
            self._emb_comb_method: function = self._init_emb_comb_method(emb_comb_method) # how to combine semantic embeddings and individual embeddings

        self.reset_params()

    def _get_emb_from_idx(self, item_ids:torch.Tensor, emb_tbl: torch.nn.Embedding) -> torch.Tensor:
        # prevent the process from crashing if item_ids are oob for the embedding table
        idx_min, idx_max = torch.min(item_ids), torch.max(item_ids)
        if idx_min < 0 or idx_max >= emb_tbl.num_embeddings:
            print(f"!Exiting...: item_ids (min: {idx_min}, max: {idx_max}) are out of bounds for emb table with size {emb_tbl.num_embeddings}.")
            import torch.distributed as dist
            dist.destroy_process_group()
            exit()
        return emb_tbl(item_ids)
    
    def _init_emb_comb_method(self, emb_comb_method: str):
        if emb_comb_method == "sum":
            def combiner(embs1: torch.Tensor, embs2: torch.Tensor) -> torch.Tensor:
                return embs1 + embs2
        elif emb_comb_method == "FC":
            self._emb_comb_fc = torch.nn.Linear(self._item_embedding_dim * 2, self._item_embedding_dim)
            def combiner(embs1: torch.Tensor, embs2: torch.Tensor) -> torch.Tensor:
                embs = torch.cat((embs1, embs2), dim=-1)
                out = self._emb_comb_fc(embs)
                return out
        elif emb_comb_method == "MLP":
            self._emb_comb_mlp = torch.nn.Sequential(
                torch.nn.Linear(self._item_embedding_dim * 2, self._item_embedding_dim * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self._item_embedding_dim * 2, self._item_embedding_dim)
            )
            def combiner(embs1: torch.Tensor, embs2: torch.Tensor) -> torch.Tensor:
                embs = torch.cat((embs1, embs2), dim=-1)
                return self._emb_comb_mlp(embs)
        elif emb_comb_method == "scalarGate":  #TODO: or maybe just a Parameter to avoid additional linear layer?
            self._emb_comb_sg = torch.nn.Linear(self._item_embedding_dim * 2, 1)
            def combiner(embs1: torch.Tensor, embs2: torch.Tensor) -> torch.Tensor:
                embs = torch.cat((embs1, embs2), dim=-1)
                gate = torch.sigmoid(self._emb_comb_sg(embs))
                out = gate * embs1 + (1 - gate) * embs2
                return out
        elif emb_comb_method == "vectorGate":
            self._emb_comb_vg = torch.nn.Linear(self._item_embedding_dim * 2, self._item_embedding_dim)
            def combiner(embs1: torch.Tensor, embs2: torch.Tensor) -> torch.Tensor:
                embs = torch.cat((embs1, embs2), dim=-1)
                gate = torch.sigmoid(self._emb_comb_vg(embs))
                out = gate * embs1 + (1 - gate) * embs2
                return out
        elif emb_comb_method == "scalarWeight":
            self.weight_alpha = torch.nn.Parameter(torch.tensor(0.5, dtype=self._item_emb_SID.weight.dtype))
            def combiner(embs1: torch.Tensor, embs2: torch.Tensor) -> torch.Tensor:
                out = self.weight_alpha * embs1 + (1 - self.weight_alpha) * embs2
                return out
        else:
            raise ValueError(f"Unknown embedding combination method: {emb_comb_method}. "
                             f"Should choose from ['sum', 'FC', 'MLP', 'scalarGate', 'vectorGate', 'scalarWeight']")
        
        print(f"**** initialized embedding combination method: {emb_comb_method}")
        return combiner
            
    def debug_str(self) -> str:
        return f"local_SID_emb_d{self._item_embedding_dim}"

    def reset_params(self):
        for name, params in self.named_parameters():
            if '_item_emb' in name:
                print(f"Initialize {name} as truncated normal: {params.data.size()} params")
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def SID2embID(self, sid_tup):
        
        convert_coeff = [self._num_codes ** i for i in range(sid_tup.shape[-1])]
        convert_coeff.reverse()
        convert_coeff = torch.tensor(convert_coeff, device=sid_tup.device).view(-1, sid_tup.shape[-1]).unsqueeze(0)
        
        emb_id = sid_tup * convert_coeff
        emb_id = torch.sum(emb_id, dim = -1)

        return emb_id
    
    def SID2multidigits_RQsum(self, sid_tup):
        convert_coeff = [self._num_codes * i for i in range(sid_tup.shape[-1])]
        convert_coeff.reverse()
        convert_coeff = torch.tensor(convert_coeff, device=sid_tup.device).view(-1, sid_tup.shape[-1]).unsqueeze(0)

        emb_id = sid_tup + convert_coeff
        return emb_id

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        item ids: B * num_ids (e.g. 1 * 3883)
        """
        # 1. extract SIDs:
        sid_tup = None
        if self._lookup is not None:
            sid_tup = self._lookup[item_ids - 1]  # B * num_ids * e.g. 3, item_ids from 1 to 4629004 (but emb tables has an extra 0 padding row)
        else:
            #TODO: SID calculation on the fly
            raise NotImplementedError
        
        # trim to the correct number of layers
        sid_tup = sid_tup[:, :, :self._num_layers]  # B * num_ids * num_layers

        # 2. extract embeddings:
        #==== Ngram condition
        if self._emb_method == "Ngram":  
            emb_id  = self.SID2embID(sid_tup) # B * 1
            emb_id_mod = torch.remainder(emb_id, self._emb_tbl_size) # B * 1
            embs = self._get_emb_from_idx(emb_id_mod, self._item_emb_SID)
            return embs

        #==== prefixN or prefixN-indEmb or RQsum-indEmb conditions
        sum_embs = torch.zeros((*item_ids.shape, self._item_embedding_dim), 
                                dtype= self._item_emb_SID.weight.dtype,
                                device = item_ids.device) # B * num_ids * 50 (emb dim)
        
        if self._emb_method == "RQsum-indEmb":
            emb_id_tups = self.SID2multidigits_RQsum(sid_tup) # B * num_ids * num_layers
            for layer_i in range(emb_id_tups.shape[-1]):
                emb = self._get_emb_from_idx(emb_id_tups[:, :, layer_i], self._item_emb_SID) # B * num_ids * 50
                sum_embs += emb

        else: # "prefixN" conditions
            for prefix_i in range(1, self._num_layers + 1):
                emb_id = self.SID2embID(sid_tup[:, :, :prefix_i])
                emb_id_mod = torch.remainder(emb_id, self._emb_tbl_size) # B * 1
                
                embs = self._get_emb_from_idx(emb_id_mod, self._item_emb_SID)
                sum_embs += embs

        if self._scale_SID_emb:
            sum_embs = sum_embs / torch.sqrt(torch.tensor(self._num_layers, device=sum_embs.device, dtype=sum_embs.dtype))
            
        if self._emb_method in ["prefixN-indEmb", "RQsum-indEmb"]:  # add individual item embeddings as well
            ind_emb = self._get_emb_from_idx(item_ids, self._item_emb_individual)
            sum_embs = checkpoint(self._emb_comb_method, sum_embs, ind_emb, use_reentrant=False) # to help with OOM
            # sum_embs = self._emb_comb_method(sum_embs, ind_emb)
            # sum_embs += ind_emb
        
        return sum_embs

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim



class CategoricalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
