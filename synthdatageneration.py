from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from colorama import Fore

import json
from typing import List
from pydantic import BaseModel #for returning specific values
from litellm import completion
from generated_prompt import prompt_template


class Record(BaseModel):
    question: str
    answer: str

class Response(BaseModel):
    generated: List[Record]

def llm_call(data: str, num_records: int = 5) -> dict:
    stream = completion(
        # model='ollama/qwen2.5:14b', 
        model="ollama/llama3.2:1b",
        messages=[
            {"role": "user",
             "content": prompt_template(data, num_records),
             }
        ],
        stream=True,
        options={'num_predict': 2000},
        format=Response.model_json_schema(),
    )
    data = ""
    for x in stream:
        delta = x['choices'][0]["delta"]['content']
        if delta is not None:
            print(Fore.LIGHTBLUE_EX + delta, end="")
            data += delta
    return json.loads(data)




if __name__ == "__main__":
    converter = DocumentConverter()
    doc = converter.convert("/Users/karnavivek/askmyprofession/data/Mixed-Integer Optimization with Constraint Learning RP.pdf").document
    
    chunker = HybridChunker() #chuck by chuck will be good for the machine
    chunks = chunker.chunk(dl_doc=doc)

    dataset = {}
    for i, chunk in enumerate(chunks):
        print(Fore.BLUE + f"Raw Text:\n{chunk.text[:300]}")

        enriched_text = chunker.contextualize(chunk=chunk) #this adds some context of the raw text at the start
        print(Fore.GREEN + f"Contextualized Text:\n{enriched_text[:300]}")

    # Till here we simply creating the input chucks using docling, but after this we want to convert that into chat template format which llama model understands
        data = llm_call(enriched_text)
        
        dataset[i] = {"generated":data['generated'], "context": enriched_text}

    with open("new_data/CLdata.json",'w') as f:
        json.dump(dataset, f)