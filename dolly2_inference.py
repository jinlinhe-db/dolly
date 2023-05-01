# Databricks notebook source
# MAGIC %pip install "accelerate>=0.16.0,<1" "transformers[torch]>=4.28.1,<5" "torch>=1.13.1,<2"

# COMMAND ----------

import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# generate_text = pipeline(model="databricks/dolly-v2-7b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-7b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-7b", device_map="auto", torch_dtype=torch.bfloat16)



# COMMAND ----------

# MAGIC %run ./summarize_pipeline

# COMMAND ----------

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)


# COMMAND ----------

text_to_summarize = "Blockchain company Ripple sold more than $361 million worth of XRP tokens in the first three months of the year, up from $226.31 million in the previous quarter, it said in its latest XRP Markets Report.\
The company builds global payment products and has developed the XRP payment system, which it describes as decentralized. While Ripple and XRP are separate entities, Ripple uses XRP and XRP's public blockchain to power its products.\
The sales were in connection with Ripple’s on-demand liquidity product, which helps customers to move money around the world without the need for correspondent banking relationships."

print(len((text_to_summarize*14).split(' ')))

# COMMAND ----------

# Put text to summarize in quotes follow by the number of words you want it to summarize to:
# Example: "bla bla bla...->5"
generate_text(f"{text_to_summarize*14}->15")

# COMMAND ----------

import torch
torch.cuda.empty_cache()

# COMMAND ----------

print(torch.cuda.memory_summary(device=None, abbreviated=False))


# COMMAND ----------


