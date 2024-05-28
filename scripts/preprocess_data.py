from datasets import load_dataset


SAMPLE_TEMPLATE="""### Instruction:
Your task is to generate valid SQL query to answer the following question, given a database schema.

### Input:
Here is the database schema that the SQL query will run on:
{database_schema}

### Question:
{question}

### Response:
{query}
"""



def generate_dataset():
    dataset = load_dataset("motherduckdb/duckdb-text2sql-25k")
    prompt_list = []
    for sample in dataset['train']:
        prompt_list.append(SAMPLE_TEMPLATE.format(database_schema=sample['schema'],
                                                  question=sample['prompt'],
                                                  query=sample['query']
                                                  ))
    
    
    samples = "\n\n".join(prompt_list)
    
    with open("../data/preprocessed_data.txt","w+") as f:
        f.write(samples)
    
    
if __name__ == "__main__":
    generate_dataset()