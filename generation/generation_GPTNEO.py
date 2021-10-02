from happytransformer import HappyGeneration
from happytransformer import GENSettings
from happytransformer import GENTrainArgs
import pandas as pd



#Training
gpt_neo = HappyGeneration(model_type="GPT-NEO", model_name="EleutherAI/gpt-neo-125M") 

#gpt_neo = HappyGeneration("GPT-Neo", "EleutherAI/gpt-neo-125M") 

train_args = GENTrainArgs(num_train_epochs=1, learning_rate=2e-05, batch_size=1) 

data = pd.read_csv("data/tripadvisor_hotel_reviews.csv")
text = ". ".join(list(data["Review"][0:50]))
text_file = open("sample.txt", "w")
n = text_file.write(text)
text_file.close()
test_set  = list(data["Review"])[-1]
gpt_neo.train("sample.txt", args=train_args)


#Generation 
top_k_sampling_settings = GENSettings(do_sample=True, top_k=50, max_length=30, min_length=10)
output_top_k_sampling = gpt_neo.generate_text("Hotel ", args=top_k_sampling_settings)

print(output_top_k_sampling.text)
