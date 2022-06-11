from transformers import AutoTokenizer, AutoModel, pipeline
import torch

class FeatureExtractor():
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", token_size=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.token_size = token_size
        self.pipeline = self.make_pipeline()

    def make_pipeline(self):
        def cut_token(token):
            for key in token:
                token[key] = token[key][:, :self.token_size]
            return token

        def tokenizer_with_cut(text, return_tensors=None):
            return cut_token(self.tokenizer(text, return_tensors=return_tensors))

        return pipeline('feature-extraction', tokenizer=tokenizer_with_cut, model=self.model)

    def extract_features(self, text):
        return torch.FloatTensor(self.pipeline(text))[0, 0, :]

if __name__ == '__main__':
    fe = FeatureExtractor()
    embedding = fe.extract_features("Территории возле станций БКЛ «Сокольники», «Рижская» и «Марьина Роща» благоустроят")
    print(embedding.shape)
    print(embedding)
