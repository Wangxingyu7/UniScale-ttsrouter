from pathlib import Path
import os
import jsonlines
from torch.utils.data import Dataset


def get_train_test_dataset(env_name, *args, **kwargs):
    env_dir = Path(__file__).parent
    train_ds = None
    if env_name == 'input':
        print(f'Loading input dataset')
        input_override = os.environ.get('TTS_INPUT_PATH')
        if input_override:
            test_ds_path = Path(input_override)
        else:
            test_ds_path = env_dir / "dataset/input.jsonl"
        test_ds = JsonlMathDataset(test_ds_path)
    elif env_name == 'MATH':
        print(f'Loading MATH dataset')
        test_ds = JsonlMathDataset(env_dir / "dataset/test500.jsonl")
    elif env_name == 'AMC23':
        print(f'Loading AMC23 dataset')
        test_ds = JsonlMathDataset(env_dir / "dataset/test_amc.jsonl")
    elif env_name == 'AIME24':
        print(f'Loading AIME24 dataset')
        test_ds = JsonlMathDataset(env_dir / "dataset/test_aime.jsonl")
    elif env_name == 'AMC23_t1':
        print(f'Loading AMC23_t1 dataset')
        test_ds = JsonlMathDataset(env_dir / "dataset/test_amc_t1.jsonl")
    elif env_name == 'gpqa_diamond':
        print(f'Loading gpqa_diamond dataset')
        test_ds = JsonlMathDataset(env_dir / "dataset/gpqa_diamond.jsonl")
    # 未适配
    elif env_name == 'ZebraLogic_grid':
        print(f'Loading ZebraLogic_grid dataset')
        test_ds = JsonlMathDataset(env_dir / "dataset/ZebraLogic_grid.jsonl")
    elif env_name == 'ZebraLogic_mc':
        print(f'Loading ZebraLogic_mc dataset')
        test_ds = JsonlMathDataset(env_dir / "dataset/ZebraLogic_mc.jsonl")
    elif env_name == 'livecodebench_v6':
        print(f'Loading livecodebench_v6 dataset')
        test_ds = JsonlMathDataset(env_dir / "dataset/livecodebench_v6.jsonl")
    return train_ds, test_ds


class JsonlMathDataset(Dataset):

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data = []
        with jsonlines.open(data_path, "r") as reader:
            for obj in reader:
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        if 'amc' in self.data_path.stem:
            item = {"question": x["problem"], "answer": str(x["solution"]), "extracted_groundtruth": str(x["extracted_groundtruth"])}
        elif 'aime' in self.data_path.stem:
            item = {"question": x["problem"], "answer": str(x["solution"]), "extracted_groundtruth": str(x["extracted_groundtruth"])}
        else:
            item = {"question": x["problem"], "answer": x["solution"], "level": x.get("level")}
    
        # 透传可选字段（若存在）
        if "lm" in x:
            item["lm"] = x["lm"]
        if "lm_idx" in x:
            item["lm_idx"] = x["lm_idx"]
        if "beam" in x:
            item["beam"] = x["beam"]
        return item
