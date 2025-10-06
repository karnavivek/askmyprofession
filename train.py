from datasets import load_dataset
from colorama import Fore

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import torch

<<<<<<< HEAD
dataset = load_dataset("/workspace/final_data", split='train')
=======
dataset = load_dataset("new_data", split='train')
>>>>>>> 36c98d41abc18e6f8ad90f867614c19de1af3cb2
print(Fore.YELLOW + str(dataset[2]) + Fore.RESET)

'''
Now we would want to convert this Q&A template into something a LLM Model can understand
i.e. we need to tokenize this & into the format that abids by our base model that is llama 3.2 1B
'''
def format_chat_template(batch, tokenizer):
    '''
    batch -> batch of training data
    tokenizer -> define what your kind of tokenizer
    '''
    system_prompt = """You are a very honest & hardworking assistant that is designed to help data science
    and operations research engineers. Think nicely & answer the question, dont make things up, dont hallucinate, if you are unsure about something
    just say you dont know the answer or you are not sure about it"""
    
    tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + " \
    "message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + " \
    "content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    samples = []

    #taking questions and answers from our func input (batch)
    questions = batch['question']
    answers = batch['answer']

    #looping through the batch to create a llama chat template for each question and answer
    for i in range(len(questions)):
        data_json = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": questions[i]},
            {"role": "assistant", "content": answers[i]}
        ]

        text = tokenizer.apply_chat_template(data_json, tokenize=False)
        samples.append(text)

    return {
        "instructions": questions,
        "response": answers,
        "text": samples
    }

base_model = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True,
<<<<<<< HEAD
    token = "add_your_own_hf_token",
=======
    # token = 
>>>>>>> 36c98d41abc18e6f8ad90f867614c19de1af3cb2
)

train_dataset = dataset.map(lambda x: format_chat_template(x, tokenizer), num_proc=8, batched=True, batch_size=10)
print(Fore.BLUE + str(train_dataset[0]) + Fore.RESET)

# --------------- After Tokenizing, Now we are going to make the model --------------------

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)


'''
Observation:
major issue here is that we cant use bitsandbytes quantization with MPS, because the 
core of "bitsandbytes" is written in CUDA i.e. Exclusive to nVIDIA & not Apple Silicon

There are 2 ways to move forward, either rely on MPS & ditch quantizatio Approach (or)
use Apple's CPU & then use bitsandbytes quantization

<<<<<<< HEAD
here is some Comparison between these approaches:
MPS + No 4bit Quant = 5 to 6 hrs
CPU + 4bit Quant    =
Cloud GPU + 4bit quantization = 25secs
=======
here is some Comparison between these 2 approaches:
MPS + No 4bit Quant = 5 to 6 hrs
CPU + 4bit Quant    =
>>>>>>> 36c98d41abc18e6f8ad90f867614c19de1af3cb2
'''


model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map = "auto",
<<<<<<< HEAD
    token="add_your_own_hf_token",
    quantization_config = quantization_config,
    cache_dir = "/workspace"
)

print(Fore.LIGHTMAGENTA_EX + str(model) + Fore.RESET)
print(Fore.CYAN + str(next(model.parameters()).device) + Fore.RESET)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    args=SFTConfig(output_dir="meta-llama/Llama-3.2-1B-Super-FT", num_train_epochs=50),
    peft_config=peft_config,
)

trainer.train()

trainer.save_model('final_opc_checkpoint')
trainer.model.save_pretrained("final_opc_model")
=======
    quantization_config = quantization_config,
    cache_dir = "/Users/karnavivek/askmyprofession/cache"

)

print(Fore.ORANGE + str(model) + Fore.RESET)
print(Fore.CYAN + str(next(model.parameters()).device) + Fore.RESET)


>>>>>>> 36c98d41abc18e6f8ad90f867614c19de1af3cb2





