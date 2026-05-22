---
dataset_info:
- config_name: grid_mode
  features:
  - name: id
    dtype: string
  - name: size
    dtype: string
  - name: puzzle
    dtype: string
  - name: solution
    struct:
    - name: header
      sequence: string
    - name: rows
      sequence:
        sequence: string
  - name: created_at
    dtype: string
  splits:
  - name: test
    num_bytes: 1545275
    num_examples: 1000
  download_size: 345826
  dataset_size: 1545275
- config_name: mc_mode
  features:
  - name: id
    dtype: string
  - name: puzzle
    dtype: string
  - name: question
    dtype: string
  - name: choices
    sequence: string
  - name: answer
    dtype: string
  - name: created_at
    dtype: string
  splits:
  - name: test
    num_bytes: 5039993
    num_examples: 3259
  download_size: 826292
  dataset_size: 5039993
configs:
- config_name: grid_mode
  data_files:
  - split: test
    path: grid_mode/test-*
- config_name: mc_mode
  data_files:
  - split: test
    path: mc_mode/test-*
---



Paper: https://huggingface.co/papers/2502.01100 

Arxiv: https://arxiv.org/abs/2502.01100

### Citation
```bibtex
@article{zebralogic2025,
    title={ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning},
    author={Bill Yuchen Lin and Ronan Le Bras and Kyle Richardson and Ashish Sabharwal and Radha Poovendran and Peter Clark and Yejin Choi},
    year={2025},
    url={https://arxiv.org/abs/2502.01100},
}


@article{dziri2024faith,
  title={Faith and fate: Limits of transformers on compositionality},
  author={Nouha Dziri and Ximing Lu and Melanie Sclar and Xiang Lorraine Li and Liwei Jian and Bill Yuchen Lin and Peter West and Chandra Bhagavatula and Ronan Le Bras and Jena D. Hwang and Soumya Sanyal and Sean Welleck and Xiang Ren and Allyson Ettinger and Za{\"i}d Harchaoui and Yejin Choi},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}

```