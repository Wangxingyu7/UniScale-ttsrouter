import copy
import torch

@torch.inference_mode()
def _skywork_infer_fn(qa_pairs: str, model, tokenizer, device, step_tag_id, step_tag='\n', special_tag_id=151652):
    rewards = []
    for qa_pair in qa_pairs:
        question, answer = qa_pair[0], qa_pair[1]
        answer = answer.replace(step_tag, f"<|vision_start|>") + f"<|vision_start|>"

        prompt_ids = tokenizer.encode(tokenizer.bos_token + question + step_tag, return_tensors="pt").squeeze(0).to(device)
        response_ids = tokenizer.encode(answer, return_tensors="pt").squeeze(0).to(device)
        indices = torch.where(response_ids == special_tag_id)
        response_ids[indices] = step_tag_id
        input_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(device)

        _, _, scores = model(input_ids=input_ids, return_probs=True)
        mask = indices[0] + len(prompt_ids)
        step_scores = scores[0][mask]
        rewards.append(copy.deepcopy(step_scores))

    del input_ids, indices, scores, prompt_ids, response_ids, mask
    torch.cuda.empty_cache()

    return rewards

@torch.inference_mode()
def _qwen_infer_fn(conversations: str, model, tokenizer, device, special_tag_id=151651, verbose=False):
    rewards = []
    for conversation in conversations:
        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(device)
        indices = (input_ids == special_tag_id)

        logits = model(input_ids)[0]
        scores = logits.softmax(dim=-1)[0]
        probabilities = scores * indices[0].unsqueeze(-1)
        step_scores = probabilities[probabilities != 0].view(-1, 2)[:, 1]
        rewards.append(copy.deepcopy(step_scores))

    del input_ids, logits, scores
    torch.cuda.empty_cache()

    return rewards

