from transformers import AutoTokenizer, AutoModel, pipeline
import torch

class FeatureExtractor:
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", token_size=512, device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.token_size = token_size
        self.device = device
        self.pipeline = self.make_pipeline()

    def make_pipeline(self):
        def cut_token(token):
            for key in token:
                token[key] = token[key][:, :self.token_size]
            return token

        def tokenizer_with_cut(text, return_tensors=None):
            return cut_token(self.tokenizer(text, return_tensors=return_tensors))

        return pipeline('feature-extraction', tokenizer=tokenizer_with_cut, model=self.model, device=0)

    def extract_features(self, text):
        result = torch.FloatTensor(self.pipeline(text))[0, 0, :]
        # print(result.shape)
        return result


if __name__ == '__main__':
    fe = FeatureExtractor()
    embedding = fe.extract_features("Территории возле станций БКЛ «Сокольники», «Рижская» и «Марьина Роща» благоустроят")
    print(embedding.shape)
    print(embedding)
