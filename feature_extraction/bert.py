from torch import Tensor
from transformers import AutoTokenizer, AutoModel, pipeline
import torch


class BertFeatureExtractor:
    """
    Класс для извлечения BERT embeddings из текста
    Args:
        model_name: Название модели трансформера для извлечения эмбеддингов
        token_size: Кол-во токенов, которое будет передаваться модели для дальнейшего 
        извлечения эмбеддингов
    """
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", token_size=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.token_size = token_size
        self.pipeline = self.make_pipeline()

    def make_pipeline(self):
        """
        Метод класса, собирающий токенайзер и модель в общий пайплайн
        """
        def cut_token(token):
            """
            Функция, обрезающая токены до размера token_size
            """
            for key in token:
                token[key] = token[key][:, :self.token_size]
            return token

        def tokenizer_with_cut(text, return_tensors=None):
            """
            Функция, возвращающая токенайзер для пайплайна
            """
            return cut_token(self.tokenizer(text, return_tensors=return_tensors))

        if torch.cuda.is_available():
            return pipeline('feature-extraction', tokenizer=tokenizer_with_cut, model=self.model, device=0)
        return pipeline('feature-extraction', tokenizer=tokenizer_with_cut, model=self.model)

    def extract_features(self, text: str) -> Tensor:
        """
        Метод класса, извлекающий BERT признаки из текста
        Args:
            text: Текст для извлечения признаков
        """
        return torch.FloatTensor(self.pipeline(text))[0, 0, :]


if __name__ == '__main__':
    fe = BertFeatureExtractor()
    embedding = fe.extract_features("Территории возле станций БКЛ «Сокольники», «Рижская» и «Марьина Роща» благоустроят")
    print(embedding.shape)
    print(embedding)
