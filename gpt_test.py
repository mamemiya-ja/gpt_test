import pandas as pd
import openai
import csv
import time

df = pd.read_csv("", delimiter='\t')
df = df[['sentence_A_Ja', 'sentence_B_Ja', 'entailment_label_Ja']]
mapping = {
        'neutral': 2,
        'contradiction': 0,
        'entailment': 1
    }
df.entailment_label_Ja = df.entailment_label_Ja.map(mapping)
df

# データの抽出
#仮説
t1_test = df.sentence_B_Ja.values
#前提
t2_test = df.sentence_A_Ja.values
labels_test = df.entailment_label_Ja.values

openai.api_key = "" 

with open("", "a") as f:
    writer = csv.writer(f)
    for i, (premise, hypothesis, label) in enumerate(zip(t2_test, t1_test, labels_test)):
        p = premise.replace(" ", "")
        h = hypothesis.replace(" ", "")
        prompt = "前提と仮定の関係は「含意」・「中立」・「矛盾」のうちどれですか。\n前提：{premise}\n仮定：{hypothesis}".format(premise=p, hypothesis=h)
        #print(prompt)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}, #※1後述
                ]
            )
        except:
            time.sleep(1800)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}, #※1後述
                ]
            )
        #print(response["choices"][0]["message"]["content"]) #返信のみを出力
        writer.writerow([premise, hypothesis, label, response["choices"][0]["message"]["content"]])
        #if i == 99:
        #    break
        
