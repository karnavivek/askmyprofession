import json

preprocessed_data = []
with open('new_data/CLdata.json','r') as f:
    data = json.load(f)
    for i, chunk in data.items():
        for pairs in chunk['generated']:
            question, answer = pairs['question'], pairs['answer']
            context_pair = {
                "question": f'{pairs['question']}',
                "answer":   pairs['answer']
            }
            preprocessed_data.append(context_pair)
            print(str(chunk))
        
        # pp_data = preprocessed_data.append(data_gen)
        # print(pp_data)

with open('new_data/preprocessed_CLdata.json','w') as f:
    json.dump(preprocessed_data,f)



# if __name__ == "__main__":
    # data_gen = data.items()
    # print(data_gen['196'])
