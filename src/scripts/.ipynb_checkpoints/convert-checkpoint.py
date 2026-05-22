#!/usr/bin/env python3
# input is a input.jsonl every single step:
# input ={"problem": {"problem": "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.", "solution": "$\\frac{9}{s} + t = 4$ in hours and $\\frac{9}{s+2} + t = 2.4$ in hours.\nSubtracting the second equation from the first, we get, \n$\\frac{9}{s} - \\frac{9}{s+2} = 1.6$\nMultiplying by $(s)(s+2)$, we get \n$9s+18-9s=18=1.6s^{2} + 3.2s$\nMultiplying by 5/2 on both sides, we get\n$0 = 4s^{2} + 8s - 45$\nFactoring gives us \n$(2s-5)(2s+9) = 0$, of which the solution we want is $s=2.5$.\nSubstituting this back to the first equation, we can find that $t = 0.4$ hours.\nLastly, $s + \\frac{1}{2} = 3$ kilometers per hour, so\n$\\frac{9}{3} + 0.4 = 3.4$ hours, or $\\framebox{204}$ minutes\n-Failure.net\nThe amount of hours spent while walking on the first travel is $\\frac{240-t}{6}$. Thus, we have the equation $(240-t)(s) = 540$, and by the same logic, the second equation yields $(144-t)(s+2) = 540$. We have $240s-st = 540$, and $288+144s-2t-st = 540$. We subtract the two equations to get $96s+2t-288 = 0$, so we have $48s+t = 144$, so $t = 144-48s$, and now we have $(96+48s)(s) = 540$. The numerator of $s$ must evenly divide 540, however, $s$ must be less than 3. We can guess that $s = 2.5$. Now, $2.5+0.5 = 3$. Taking $\\frac{9}{3} = 3$, we find that it will take three hours for the 9 kilometers to be traveled. The t minutes spent at the coffeeshop can be written as $144-48(2.5)$, so t = 24. $180 + 24 = 204$. -sepehr2010", "extracted_groundtruth": "204", "url": "https://artofproblemsolving.com/wiki/index.php/2024_AIME_I_Problems/Problem_1 "}, "QP": 2, "CP": 4, "BS": 1, "model": "/root/autodl-pub/policy_models/Qwen3-0.6B"}
# or
# input={"problem": "Every morning Aya goes for a 9-kilometer-long walk ...", "QP": 2, "CP": 4, "BS": 1, "model": "/root/autodl-pub/policy_models/Qwen3-0.6B"}



import json
import sys

def enrich_one(item: dict) -> dict:
    
    new_item = item.copy()
    for k in ("QP", "CP", "BS", "model"):
        new_item[k] = item[k]
    return new_item

def main(input_path='input.jsonl'):
    with open(input_path,  'r', encoding='utf-8') as input_file, \
         open('output.jsonl', 'w', encoding='utf-8') as output_file:

        for line in input_file:
            line = line.strip()
            if not line:
                continue
            try:
                raw_record = json.loads(line)
                enriched_record = enrich_one(raw_record)
                output_file.write(json.dumps(enriched_record, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as err:
                print(f'[WARN] 跳过一行：{err}', file=sys.stderr)

    print('已写入 output.jsonl', file=sys.stderr)

if __name__ == '__main__':
    main()