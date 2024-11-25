import pandas as pd
import time
import re

import goodfire

client = goodfire.Client(
    "sk-goodfire-KDDIRyJ7fB2QwxFY7PDEfL2Uo4793Rfw40mRnwEK6OZeiXmn8HXqZw"
  )

# Instantiate a model variant
llama_8b = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")
llama_70b = goodfire.Variant("meta-llama/Meta-Llama-3.1-70B-Instruct")

math_features_1, relevance = client.features.search("question words quantites", model=llama_70b, top_k=5)
math_features_2, relevance = client.features.search("math problem", model=llama_70b, top_k=5)

llama_70b.set(math_features_1[0], 0.3)
#llama_70b.set(math_features_2[0], 0.1)

print(llama_70b)


multi_df = pd.read_csv('multilingual_arith_gsm100.csv')
temp_df = pd.read_csv('multi_prompt_template.csv')

languages = ["English", "Hindi", "French", "Italian", "Spanish", "Portugese", "German", "Thai"]
ans_dictionary = {}

for lang in languages:
  resp = lang + "_resp"
  ans = lang + "_ans"
  ans_dictionary[resp] = []
  ans_dictionary[ans] = []


count = 0
max_retries = 10
retry_count = 0
for index, row in multi_df[5:25].iterrows():
  count += 1
  print(f"processing index {count}")
  for language in languages:
    #print(language)
    #print(f"system prompt = {temp_df[language][1]}")
    #print(f"question = {row[language]}")
    #print(f"suffix prompt = {temp_df[language][0]}")
    system_prompt = temp_df[language][1]
    question = row[language]
    suffix_prompt = temp_df[language][0]
    

    while retry_count < max_retries:
      try:
        resp = client.chat.completions.create(
        [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": question + suffix_prompt },
        ],
        model=llama_70b,
        stream=False,
        max_completion_tokens=1000,
        )
        retry_count = 0
        break
      except Exception as e:
        retry_count += 1
        sleep_time = 2 ** retry_count
        print(f"exception seen while processing {language}")
        print(f"Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    model_response = resp.choices[0].message['content']

    #mth = re.search(r"####\s*(\d+)", model_response)
    mth = re.search(r"####\s*[^0-9]*?([\d,]+)", model_response)
    
    if mth:
      try:
          final_answer = int(mth.group(1).replace(',', ''))
      except Exception as e:
          final_answer = -1
    else:
      final_answer = -1

    ans_dictionary[language + "_resp"].append(model_response)
    ans_dictionary[language + "_ans"].append(final_answer)
    #print(f"response = {model_response}")
    #print(f"numerical_answer = {final_answer}")
    #print("=================================") 



df = pd.DataFrame(ans_dictionary)
df.to_csv("variant_ans_2.csv", index=False)
