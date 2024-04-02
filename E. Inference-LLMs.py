import os
import json
import torch

from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, StoppingCriteria, StoppingCriteriaList

@dataclass
class GenerationArguments:
    temperature: float = field(default=0.5)
    top_p: float = field(default=0.0)
    top_k: int = field(default=0)
    max_new_tokens: int = field(default=512)
    early_stopping: bool = field(default=False)
    do_sample: bool = field(default=False)
    

@dataclass
class ModelArguments:
    model_name: str
    model_dir: str = field(default="results/")
    


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.unsqueeze(0).to("cuda") if stop.dim() == 0 else stop.to("cuda")  for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


def get_stopping_criteria(tokenizer):
    stop_words = ["<|im_end|>", "<end_of_turn>", "user", "system ", "<<SYS>>", "[INST]", "\n\n"]
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    return stopping_criteria


def test():
    parser = HfArgumentParser((ModelArguments, GenerationArguments))
    model_args, gen_args = parser.parse_args_into_dataclasses()
    
    model_args.model_name = os.path.join(model_args.model_dir, model_args.model_name)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if "gemma" in model_args.model_name:
        source_prompt = """<|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>"""
        if "it" in model_args.model_name:
            source_prompt = """<start_of_turn>user\n{system_content}\n{user_content}<end_of_turn>"""
    elif "llama" in model_args.model_name:
        source_prompt = """[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{user_content} [/INST]"""
    elif "mistral" in model_args.model_name:
        source_prompt = """[INST]{system_content}\n{user_content} [\INST]"""
    else:
        raise "Choose a model from gemma & llama2 & mistral"

    # stop words
    stopping_criteria = get_stopping_criteria(tokenizer)
    # data
    with open("inference.json", "r") as f:
        test_data = json.load(f)

    with open(model_args.model_name + '-test.txt', 'w') as f:
        for data in tqdm(test_data):
            # Prompt 적용
            input_text = source_prompt.format_map(data)
            input_tensor = tokenizer(
                    input_text,
                    return_tensors='pt',
                    max_length=3500,
                    truncation=True,
                    padding="longest"
                ).to('cuda')
            
            start_len = input_tensor.input_ids.shape[-1]
            output = model.generate(
                **input_tensor,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=gen_args.max_new_tokens,
                stopping_criteria=stopping_criteria
            )

            result = tokenizer.decode(output[0][start_len + 1:])
            result = tokenizer.decode(output[0][start_len:])
            matched = re.match("<\|im_start\|>assistant\s+|<start_of_turn>model\s+", result)
            if matched:
                result = result[matched.span()[-1]:]
                
            f.write("-" * 20 + "\n")
            f.write("Label\n")
            f.write(data['assistant_content'] + '\n')
            f.write("Pred\n")
            f.write(result + '\n')
            f.write("-" * 20 + '\n')


if __name__ == "__main__":
    test()
