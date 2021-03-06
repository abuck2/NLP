#from transformers import pipeline
import sklearn
import numpy as np
import pandas as pd
from transformers import pipeline
from aitextgen import aitextgen
from aitextgen.colab import mount_gdrive, copy_file_from_gdrive
import os


class ReviewGeneratorGPT:
    def __init__(self):
        #https://becominghuman.ai/text-generation-using-gpt3-781429c4169
        #https://www.kaggle.com/jdparsons/gpt-2-fake-real-disasters-data-augmentation
        #https://www.tensorflow.org/text/tutorials/text_generation
        print("initalize")
        self.generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

    def run(self):
        one_sentence = "nice hotel expensive parking got good deal stay hotel anniversary, arrived late evening took advice previous reviews did valet parking, check quick easy, little disappointed non-existent view room room clean nice size, bed comfortable woke stiff neck high pillows, not soundproof like heard music room night morning loud bangs doors opening closing hear people talking hallway, maybe just noisy neighbors, aveda bath products nice, did not goldfish stay nice touch taken advantage staying longer, location great walking distance shopping, overall nice experience having pay 40 parking night"
        """
        #input_ids = tokenizer.encode(one_sentence, return_tensors = "pt")
        output = model.generate(input_ids,
            max_length = 10000,
            num_beams = 5,
            no_repeat_ngram_size  = 2,
            early_stopping = True)


        result = tokenizer.decode(output[0], skip_special_tokens = True)
        """
        result = generator(one_sentence, max_length=100, do_sample=True, temperature=0.9)
        print(result)

class ReviewGeneratorGPTNeo:
    def __init__(self, small:bool = True):
        #https://colab.research.google.com/drive/15qBZx5y9rdaQSyWpsreMDnTiZ5IlN0zD?usp=sharing#scrollTo=KBkpRgBCBS2_
        print("initalize")
        if small : 
            self.ai = aitextgen(model="EleutherAI/gpt-neo-125M", to_gpu=True)
        else : 
            self.ai = aitextgen(model="EleutherAI/gpt-neo-355M", to_gpu=True)
        self.current_dir =  os.path.dirname(os.path.abspath(__file__))
        data = pd.read_csv(self.current_dir+"/data/tripadvisor_hotel_reviews.csv")
        data["Review"].to_csv(self.current_dir+"/data/single_col.csv")
    
    def run(self):
        one_sentence = "nice hotel expensive parking got good deal stay hotel anniversary, arrived late evening took advice previous reviews did valet parking, check quick easy, little disappointed non-existent view room room clean nice size, bed comfortable woke stiff neck high pillows, not soundproof like heard music room night morning loud bangs doors opening closing hear people talking hallway, maybe just noisy neighbors, aveda bath products nice, did not goldfish stay nice touch taken advantage staying longer, location great walking distance shopping, overall nice experience having pay 40 parking night"
        self.ai.train(self.current_dir+"/data/single_col.csv",
            line_by_line=True,
            from_cache=False,
            num_steps=3000,
            generate_every=1000,
            save_every=1000,
            save_gdrive=False,
            learning_rate=1e-3,
            fp16=False,
            batch_size=1, 
            )

        ai.generate()

if __name__=="__main__":
    rg = ReviewGeneratorGPTNeo(small = True)
    rg.run()
