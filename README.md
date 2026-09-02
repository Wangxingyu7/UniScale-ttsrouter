# ttsrouter-v1.1


## Getting Started

### Installation

Clone the repository:

```bash
git clone <ANONYMOUS_REPOSITORY_URL>
cd ttsrouter-v1.1/src
```

Create a clean environment:

```bash
conda create -n tts python=3.12
conda activate tts
```

We recommend using .whl to install most dependencies.You can download it at [vllm/releases](https://github.com/vllm-project/vllm/releases/download/v0.11.2/vllm-0.11.2-cp38-abi3-manylinux1_x86_64.whl).

Install the prebuilt wheel first (fast and reliable):

```bash
# example: place the wheel in the src
cd src
pip install vllm-0.11.2-cp38-abi3-manylinux1_x86_64.whl
```

Then install the rest of the dependencies:

```bash
pip install --upgrade-strategy=eager -r requirements.txt
```

Install `tmux` for serving policy models and PRMs:

```bash
sudo apt-get update
sudo apt-get install tmux
```

Install `yq` for processing yaml files:

```bash
mkdir -p "$CONDA_PREFIX/bin"
wget -qO "$CONDA_PREFIX/bin/yq" \
  https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
chmod +x "$CONDA_PREFIX/bin/yq"
```

### Supported Tasks

- [MATH-500](https://github.com/openai/prm800k)
- [AIME24](https://huggingface.co/datasets/AI-MO/aimo-validation-aime)
- [AMC23](https://huggingface.co/datasets/AI-MO/aimo-validation-amc)

### Supported Models

#### Policy Models

Qwen series :

- [Qwen3](https://huggingface.co/collections/Qwen/qwen3): [0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), [4B](https://huggingface.co/Qwen/Qwen3-4B), [8B](https://huggingface.co/Qwen/Qwen3-8B), [14B](https://huggingface.co/Qwen/Qwen3-14B), [32B](https://huggingface.co/Qwen/Qwen3-32B)

#### Process Reward Models

> **Note:**
> For Skywork-PRM, support for vllm >= 0.6.4.post1 is no longer planned, so we need to modify some parts of the vllm source code to support Skywork-PRM, following the steps below:

First, locate the source code:

```bash
python -c "import vllm, inspect, pathlib; print(pathlib.Path(inspect.getfile(vllm)).parent)"
```

Secondly, apply the patch in `src/vllmchange.md` to the vLLM source code.

- [Skywork](https://huggingface.co/collections/Skywork/skywork-o1-open-67453df58e12f6c3934738d0): [Skywork-PRM-1.5B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B), [Skywork-PRM-7B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B)
- [Qwen2.5-Math](https://huggingface.co/collections/Qwen/qwen25-math-66eaa240a1b7d5ee65f1da3e): [Qwen2.5-Math-PRM-7B](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B)

For Qwen-PRM, we don't need to modify the vllm source code. However, since the step token of Qwen-PRM is <extro_0> and differs from Skywork's step token("\n"), if you want to use Qwen-PRM, you need to modify the logic in src/reason/vllm_rm_infer_fns.py, or switch to using infer_fns.py.

### GPU configurations (recommended)

| Policy Model                  | PRM   | GPU            | CUDA |
|-------------------------------|-------|----------------|------|
| 0.6B, 1.7B, 4B, 8B, 14B       | 1.5B  | 4x RTX5090 32GB| 12.8 |
| 32B                           | 1.5B  | 2x L40 44GB    | 13.0 |
| 0.6B, 1.7B, 4B, 8B, 14B, 32B  | 1.5B  | 4x A100 80GB   | 13.0 |

### How to run

We offer two modes of operation:  

1) **Development Mode**: Manually performing two stages via a script—deploying the model and starting inference—suitable for the project testing phase.
They are two mutually independent ways.
2) **Production Mode**: Deploying an interface through FastAPI for upper-level calls, suitable for when the project is fully mature;  

#### Development Mode

##### Step 1: Serve policy models and PRMs

Set the environment variables:

```bash
cd src
export VALUE_MODEL_PATH=<YOUR_PRM_PATH>/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
export POLICY_MODEL_1_PATH=<YOUR_POLICY_PATH>/Qwen3-0.6B
export POLICY_MODEL_2_PATH=<YOUR_POLICY_PATH>/Qwen3-1.7B
export LOGDIR="${PWD}/logs_vllm"
export HOST_ADDR=0.0.0.0
export CONTROLLER_PORT=21001
export WORKER_BASE_PORT=10081
```

Start the script to deploy the LM and RM models:

```bash
# 1 gpu
bash scripts/serve_gpu1_t1.sh 2 $POLICY_MODEL_1_PATH $POLICY_MODEL_2_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT
```

##### Step 2: Run TTS methods

We use beam search as the TTS method in our project, and you can extend it to other TTS methods by modifying the beam search.

###### Beam Search

```bash
cd src
bash scripts/run_t1.sh --method beam_search --RM $VALUE_MODEL_PATH --task_name AMC23_t1 
```

###### Additional Arguments

We also provide a general script for quickly deploying models at src/scripts/serve_models.sh and a general script for quickly running large-scale experiments at src/scripts/run_all_variants.sh. You can refer to their contents to understand how to use them.

#### Production Mode

##### Step1: Serve policy models and PRMs

Same to [Development Mode ➜ Step 1](#development-mode).

##### Step2: Deploy service

```bash
cd src
python -m service.serve

```

##### Step3: Send Request

###### Single Request

```bash
curl -X POST http://localhost:7777/tts-router-json \
  -H "Content-Type: application/json" \
  -d '{
    "problems": {
      "problem": "Cities $A$ and $B$ are $45$ miles apart. Alicia lives in $A$ and Beth lives in $B$. Alicia bikes towards $B$ at 18 miles per hour. Leaving at the same time, Beth bikes toward $A$ at 12 miles per hour. How many miles from City $A$ will they be when they meet?",
      "solution": "27.0",
      "lm": "Qwen3-0.6B",
      "beam": { "QP": 2.0, "CP": 8.0, "BS": 4 }
    },
    "eval_config": {
      "method": "beam_search"
    }
  }'
```

###### Mult-Request

```bash
curl -X POST http://localhost:7777/tts-router-json \
  -H "Content-Type: application/json" \
  --data-binary @- <<'JSON'
{
  "problems": [
    {
      "problem": "How many digits are in the base-ten representation of $8^5 \\cdot 5^{10} \\cdot 15^5$?",
      "solution": "18.0",
      "lm": "Qwen3-0.6B",
      "beam": { "QP": 2.0, "CP": 8.0, "BS": 4 }
    },
    {
      "problem": "Cities $A$ and $B$ are $45$ miles apart. Alicia lives in $A$ and Beth lives in $B$. Alicia bikes towards $B$ at 18 miles per hour. Leaving at the same time, Beth bikes toward $A$ at 12 miles per hour. How many miles from City $A$ will they be when they meet?",
      "solution": "27.0",
      "lm": "Qwen3-0.6B",
      "beam": { "QP": 2.0, "CP": 8.0, "BS": 4 }
    }
  ],
  "eval_config": { "method": "beam_search" }
}
JSON
```

###### Send File Request

```bash
curl -X POST http://localhost:7777/tts-router \
  -F "file=@./test.jsonl;type=application/jsonl" \
  -F "eval_config=@-;type=application/json" <<'JSON'
{"method":"beam_search"}
JSON

```

## License

This repository contains proprietary source-available project code and
third-party/open-source materials. As a whole, it is not open source.

For the original proprietary portions owned by Wangxingyu7, all rights are
reserved. No permission is granted to use, copy, modify, distribute, publish,
deploy, benchmark with, train on, submit academic work from, or create
derivative works from those proprietary portions without prior express written
authorization from the copyright holder. See [LICENSE](LICENSE).

Files under `src/envs/` and `src/utils.py` are based on or adapted from
Apache-2.0-covered open source materials and remain subject to Apache-2.0 and
applicable upstream notices. Other third-party materials remain subject to their
own licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Data files under `src/data/` are proprietary project data. They may not be used,
copied, redistributed, benchmarked with, trained on, analyzed, cited, submitted
in academic work, or used to generate experimental results without prior express
written authorization from the copyright holder. See
[DATA_USAGE.md](DATA_USAGE.md).

See [NOTICE](NOTICE) for a short repository-level proprietary notice.


