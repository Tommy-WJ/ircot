from abc import ABC, abstractmethod
from typing import Literal
import json

class MyRetrieverWrapper(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

import numpy as np
from gritlm import GritLM
import torch
from typing import List, Union, cast
import numpy as np
from tqdm import tqdm

class GritFlashAttn(GritLM):
    def __init__(self, *args, **kwargs):
        super().__init__(model_name_or_path="GritLM/GritLM-7B", attn_implementation = "flash_attention_2", torch_dtype=torch.bfloat16, *args, **kwargs)
        self.tokenizer.padding_side = "left"

    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        instruction: str = "",
        embed_instruction: bool = False,
        get_cache: bool = False,
        convert_to_tensor: bool = False,
        recast: bool = False,
        add_special_tokens: bool = True,
        use_tqdm: bool = False,
    ) -> np.ndarray:
        if not use_tqdm:
            use_tqdm = len(sentences) > batch_size
        if self.num_gpus > 1:
            batch_size *= self.num_gpus

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings, all_kv_caches = [], []
        if use_tqdm:
            iterator = tqdm(range(0, len(sentences), batch_size), desc="Batches")
        else:
            iterator = range(0, len(sentences), batch_size)
        for start_index in iterator:
            sentences_batch = [
                instruction + s + self.embed_eos for s in sentences[start_index:start_index + batch_size]
            ]
            # This will prepend the bos token if the tokenizer has `add_bos_token=True`
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
                add_special_tokens=add_special_tokens,
            ).to(self.device)

            if (self.attn is not None) and (self.attn[:2] == 'bb'):
                inputs["is_causal"] = False
            if get_cache:
                inputs['use_cache'] = True
            outputs = (
                getattr(self.model, self.embedding_attr) if self.embedding_attr else self.model
            )(**inputs)
            last_hidden_state = outputs[0]
            if get_cache:
                # Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`
                assert len(all_kv_caches) == 0, "Can only get cache for one batch at a time"
                all_kv_caches = outputs[1]

            if self.projection:
                last_hidden_state = self.projection(last_hidden_state)
            if (instruction) and (embed_instruction is False) and ("mean" in self.pooling_method):
                # Remove instruction tokens from the embeddings by masking them
                instruction_tokens = self.tokenizer(
                    instruction,
                    padding=False,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=add_special_tokens,
                )["input_ids"]

                cumsum = torch.cumsum(inputs['attention_mask'], dim=1)
                mask = cumsum <= len(instruction_tokens)
                inputs['attention_mask'][mask] = 0

            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'], recast=recast)
            # Normalize can change the dtype (https://discuss.pytorch.org/t/tensor-in-float16-is-transformed-into-float32-after-torch-norm/110891)
            if self.normalized: 
                in_dtype = embeddings.dtype
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1).to(in_dtype)
            embeddings = cast(torch.Tensor, embeddings)
            if convert_to_tensor:
                all_embeddings.append(embeddings)
            else:
                # NumPy does not support bfloat16
                all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())

        all_embeddings = (
            torch.cat(all_embeddings, dim=0) if convert_to_tensor else np.concatenate(all_embeddings, axis=0)
        )
        if input_was_string:
            all_embeddings = all_embeddings[0]
        if get_cache:
            return all_embeddings, all_kv_caches
        return all_embeddings
    
class DummyResponse:
    def __init__(self, json_data):
        self._json_data = json_data
        self.status_code = 200
        self.ok = True
    
    def json(self):
        return self._json_data

class GritWrapper(MyRetrieverWrapper):
    def __init__(self, dataset_name: Literal["hotpotqa", "musique", "2wikimultihopqa"]):
        super().__init__()
        self.dataset_name = dataset_name
        corpus_path = f"data/{dataset_name}_corpus.json"
        with open(corpus_path, "r") as f:
            corpus = json.load(f)
        self.doc_list = [f"{row['title']}\n{row['text']}" for row in corpus]
        
        # data/lm_vectors/GritLM_GritLM-7B_mean/2wikimultihopqa_doc
        embedding_dir = f"data/lm_vectors/GritLM_GritLM-7B_mean/{dataset_name}_doc"
        vec_path = f"{embedding_dir}/vecs.npy"
        doc_path = f"{embedding_dir}/encoded_strings.json"

        strings_to_encode = self.doc_list
        vectors = np.load(vec_path)
        print('Loaded {} vectors from {}'.format(len(vectors), vec_path))
        saved_strings = json.load(open(doc_path, 'r'))

        missing_strings = list(set(strings_to_encode).difference(set(saved_strings)))
        if len(missing_strings):
            print('Encoding {} Missing Strings'.format(len(missing_strings)))
            self.load_model()
            new_vectors, new_strings, = self.encode_strings_func(missing_strings)
            vectors = np.vstack([vectors, new_vectors])
            saved_strings.extend(new_strings)
            # sort by alphabet
            sorted_indices = np.argsort(saved_strings)
            vectors = vectors[sorted_indices]
            saved_strings = [saved_strings[i] for i in sorted_indices]
            np.save(vec_path, vectors)
            with open(doc_path, 'w') as f:
                json.dump(saved_strings, f)
        string_to_index = {s: idx for idx, s in enumerate(saved_strings)}
        strings_indices = [string_to_index[s] for s in strings_to_encode]
        vectors = vectors[strings_indices]
        self.embedding = vectors

        self.model = GritFlashAttn()

    def __call__(self, url, params):
        # Simulate an API request
        query_text = params.get('query_text', '')
        retrieval_count = params.get('max_hits_count', 1)
        query_embedding = self.model.encode(query_text, instruction="retrieve the most relevant documents")
        similarity_scores = np.dot(self.embedding, query_embedding)
        top_indices = np.argsort(similarity_scores)[::-1][:retrieval_count]
        retrieval = []
        for idx, doc_idx in enumerate(top_indices):
            doc = self.doc_list[doc_idx]
            retrieval_item = {
                # "title": doc.get("title", ""),
                "title": doc.split("\n")[0],
                # "paragraph_text": doc.get("paragraph_text", ""),
                "paragraph_text": doc.split("\n")[1],
                "corpus_name": self.dataset_name,
                "score": float(similarity_scores[doc_idx])
            }
            retrieval.append(retrieval_item)
        return DummyResponse({"retrieval": retrieval})
    

def get_hf_retriever(retriever_name, source_corpus_name):
    if "grit" in retriever_name.lower():
        return GritWrapper(dataset_name=source_corpus_name)
    else:
        raise ValueError(f"Retriever {retriever_name} not supported.")