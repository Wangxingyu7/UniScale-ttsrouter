
# skywork-prm-1.5b: add Qwen2ForPrmModel to vllm

## vllm/model_executor/models/qwen2_rm.py

lines 128 add:

```python
@default_pooling_type("STEP")  
class Qwen2ForPrmModel(Qwen2RewardBaseModel):
    """
    Skywork Process Reward Model adapter.
    Maps v_head.summary.* to score.* for vLLM compatibility.
    """
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        vllm_config.model_config.hf_config.num_labels = 1
        
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        
        if hasattr(self, 'score'):
            del self.score
        
        self.score = RowParallelLinear(
            config.hidden_size,
            1,  
            params_dtype=self.head_dtype,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.score",
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler(
            {"token_classify": Pooler.for_token_classify(pooler_config)}
        )


    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Robust weight mapping for Skywork PRM checkpoints.
        Handles v_head.summary.* → score.* conversion.
        """
        weight_mappings = [
            ("v_head.summary.weight", "score.weight"),
            ("v_head.summary.bias", "score.bias"),
        ]
        
        ignore_patterns = ["v_head.dropout", "lm_head."]
        
        mapped_weights = []
        
        for name, tensor in weights:
            if any(pattern in name for pattern in ignore_patterns):
                continue
            
            mapped_name = name
            for src, dst in weight_mappings:
                if src in name:
                    mapped_name = name.replace(src, dst)
                    break
            
            mapped_weights.append((mapped_name, tensor))
            
        loader = AutoWeightsLoader(self, ignore_unexpected_prefixes=["lm_head."])
        return loader.load_weights(mapped_weights)
```

## vllm/model_executor/models/registry.py

lines 206 add:

```python
    "Qwen2ForPrmModel": ("qwen2_rm", "Qwen2ForPrmModel"),
```
