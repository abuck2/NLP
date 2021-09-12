#from transformers import pipeline
import sklearn
import numpy as np
import pandas as pd
from transformers import pipeline


class ReviewGeneratorGPT:
    def __init__(self):
        #https://becominghuman.ai/text-generation-using-gpt3-781429c4169
        #https://www.kaggle.com/jdparsons/gpt-2-fake-real-disasters-data-augmentation
        #https://www.tensorflow.org/text/tutorials/text_generation
        print("initalize")
        self.generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

    def run(self):
        one_sentence = "nice hotel expensive parking got good deal stay hotel anniversary, arrived late evening took advice previous reviews did valet parking, check quick easy, little disappointed non-existent view room room clean nice size, bed comfortable woke stiff neck high pillows, not soundproof like heard music room night morning loud bangs doors opening closing hear people talking hallway, maybe just noisy neighbors, aveda bath products nice, did not goldfish stay nice touch taken advantage staying longer, location great walking distance shopping, overall nice experience having pay 40 parking night"
        input_ids = tokenizer.encode(one_sentence, return_tensors = "pt")
        output = model.generate(input_ids,
            max_length = 10000,
            num_beams = 5,
            no_repeat_ngram_size  = 2,
            early_stopping = True)


        result = tokenizer.decode(output[0], skip_special_tokens = True)

if __name__=="__main__":
    rg = ReviewGeneratorGPT()
    rg.run()
