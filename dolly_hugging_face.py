# Databricks notebook source
# MAGIC %md
# MAGIC ## Testing Dolly from Hugging Face in the notebook

# COMMAND ----------

# MAGIC %pip install accelerate>=0.12.0 transformers[torch]==4.25.1 numpy==1.21.5

# COMMAND ----------

import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v1-6b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v1-6b", device_map="auto", trust_remote_code=True, offload_folder='/Workspace/Repos/jinlin.he@databricks.com/dolly/offload')

# COMMAND ----------

!pwd

# COMMAND ----------

tokenizer.save_pretrained("/FileStore/jinlin.he@databricks.com/dolly/tokenizer/") #"/Workspace/Repos/jinlin.he@databricks.com/dolly/")

# COMMAND ----------

model.save_pretrained("/FileStore/jinlin.he@databricks.com/dolly/")

# COMMAND ----------

#model_copy = AutoModelForCausalLM.from_pretrained("/FileStore/jinlin.he@databricks.com/dolly/")


# COMMAND ----------

# import pickle
# with open('/Workspace/Repos/jinlin.he@databricks.com/dolly/AutoTokenizer.pickle', 'wb') as f:
#   pickle.dump(tokenizer, f)

# COMMAND ----------

PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def generate_response(instruction: str, *, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                      do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **kwargs) -> str:
    input_ids = tokenizer(PROMPT_FORMAT.format(instruction=instruction), return_tensors="pt").input_ids.to("cuda")

    # each of these is encoded to a single token
    response_key_token_id = tokenizer.encode("### Response:")[0]
    end_key_token_id = tokenizer.encode("### End")[0]

    gen_tokens = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id,
                                do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, **kwargs)[0].cpu()

    # find where the response begins
    response_positions = np.where(gen_tokens == response_key_token_id)[0]

    if len(response_positions) >= 0:
        response_pos = response_positions[0]
        
        # find where the response ends
        end_pos = None
        end_positions = np.where(gen_tokens == end_key_token_id)[0]
        if len(end_positions) > 0:
            end_pos = end_positions[0]

        return tokenizer.decode(gen_tokens[response_pos + 1 : end_pos]).strip()

    return None

# Sample similar to: "Excited to announce the release of Dolly, a powerful new language model from Databricks! #AI #Databricks"
generate_response("What is a stinky slinky?", model=model, tokenizer=tokenizer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap dolly into a MLflow model for model serving
# MAGIC Problem: now the tokenizer is saved as an artifact, but model is not. Model file (`model.bin`) is too big (12G) to be packed by `mlflow.pyfunc.log_model()`. So each inference call is now loading model directly from Hugging Face hub - which takes ~3 min and then does the inference.
# MAGIC
# MAGIC **Alternatives to try:**
# MAGIC  1. compress `model.bin` and hopefully it is small enough to be directly packed by `mlflow.pyfunc.log_model` as an artifact
# MAGIC  2. compress `model.bin`, upload it to DBFS, pass on the model path to `mlflow.pyfunc.log_model` and override `loader_module` 
# MAGIC  3. use `mlflow.pyfunc.save_model` to specify a path, and same as 2 (pass on the model path to `mlflow.pyfunc.log_model` and override `loader_module` )

# COMMAND ----------

import pandas as pd
df = pd.DataFrame({"Hello world?"})

# COMMAND ----------

df.iloc[:,0].to_list()[0]

# COMMAND ----------

## Wrap dolly into a MLflow model
import mlflow
import mlflow.pyfunc
from mlflow.utils.environment import _mlflow_conda_env
import numpy 
import transformers

class DollyModel(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
  
  def load_context(self, context):
    
    import pickle
    with open(context.artifacts['tokenizer'], "rb") as f:
      self.tokenizer = pickle.load(f)


  def predict(self, context, instruction_df):
    import numpy as np
    from transformers import AutoModelForCausalLM
    tokenizer = self.tokenizer
    
    #TODO: put into context?
    do_sample = True
    max_new_tokens = 256
    top_p = 0.92 
    top_k = 0

    # Parse instruction_df fed from serving end point into string
    instruction_text = instruction_df.iloc[:,0].to_list()[0]
    PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:
{instruction}
### Response:
"""

    input_ids = tokenizer(PROMPT_FORMAT.format(instruction=instruction_text), return_tensors="pt").input_ids.to("cuda")

    # each of these is encoded to a single token
    response_key_token_id = tokenizer.encode("### Response:")[0]
    end_key_token_id = tokenizer.encode("### End")[0]

    # TODO: so far can only load from hugging face
    model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v1-6b", trust_remote_code=True).to('cpu') # device_map="auto", offload_folder='/Workspace/Repos/jinlin.he@databricks.com/dolly/offload', 
    gen_tokens = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id,
                                do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k)[0].cpu()

    # find where the response begins
    response_positions = np.where(gen_tokens == response_key_token_id)[0]

    if len(response_positions) >= 0:
        response_pos = response_positions[0]
        
        # find where the response ends
        end_pos = None
        end_positions = np.where(gen_tokens == end_key_token_id)[0]
        if len(end_positions) > 0:
            end_pos = end_positions[0]

        return tokenizer.decode(gen_tokens[response_pos + 1 : end_pos]).strip()

    return None
 

with mlflow.start_run(run_name='dolly_hugging_face'):
  artifacts = {
    "tokenizer": "/Workspace/Repos/jinlin.he@databricks.com/dolly/AutoTokenizer.pickle"
  }

  ## TODO: have to compress it before hand - what is the size limit?
  #model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v1-6b", device_map="auto", trust_remote_code=True, offload_folder='/Workspace/Repos/jinlin.he@databricks.com/dolly/offload')
  #wrappedModel = DollyModel(model)
   
  # libraries_in_notebook = ["numpy", "transformers"]

  conda_env =  _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["accelerate>=0.12.0", "datasets==2.8.0", "transformers[torch]==4.25.1", "numpy==1.21.5"],
        #[f"{lib}=={eval(lib).__version__}" for lib in libraries_in_notebook],
        additional_conda_channels=None,
    )
  mlflow.pyfunc.log_model("dolly_load_from_hf",
                          python_model=DollyModel(None), 
                          conda_env=conda_env,
                          artifacts=artifacts)

# COMMAND ----------

# MAGIC %pip freeze > /Workspace/Repos/jinlin.he@databricks.com/dolly/inference/requirements.txt

# COMMAND ----------

# /FileStore/jinlin.he@databricks.com/dolly/

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test inference in the notebook

# COMMAND ----------

import mlflow 
previous_run = '738fbe7ce5e0417dbbb53a8d31cfc3df/'
logged_model = 'runs:/dbcc8e3b1d4f4feb9733bc7a8c3a04af/dolly_load_from_hf'

loaded_model = mlflow.pyfunc.load_model(logged_model)

#loaded_model.predict("What is a stinky slinky?")

# COMMAND ----------



# COMMAND ----------

import pandas as pd
loaded_model.predict(pd.DataFrame({'instruction': ["Tell me about how the universe started?"]}))


# COMMAND ----------

pd.DataFrame({'instruction': ["Tell me about how the universe started?"]})

# COMMAND ----------


