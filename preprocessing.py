import json

preprocessed_data = []
with open('new_data/CLdata.json','r') as f:
    data = json.load(f)
    for i, chunk in data.items():
        data_gen = chunk['generated']
        preprocessed_data.append(data_gen)
        # print(data_gen)

with open('new_data/preprocessed_CLdata.json','w') as f:
    json.dump(preprocessed_data.append(data_gen),f)



# if __name__ == "__main__":
    # data_gen = data.items()
    # print(data_gen['196'])
