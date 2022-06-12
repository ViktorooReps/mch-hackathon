from typing import Iterable, List, Tuple, Optional, Set, NamedTuple, Callable

import nltk.tokenize
import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from torch import Tensor

from feature_extraction.bert import BertFeatureExtractor
from feature_extraction.sequence_matcher.levenstein import match


class TextChunk(NamedTuple):
    relative_position: int
    text_position: int
    text: str


class MatchingResult(NamedTuple):
    source: Optional[TextChunk]
    target: Optional[TextChunk]


def semantic_match(
        feature_extractor: Callable[[str], Tensor],
        source_text_chunks: Iterable[str],
        target_text_chunks: Iterable[str],
        *,
        match_n_closest: int = 3,
        matching_confidence: float = 0.5,
        positional_weight: float = 0.3,
        matching_weight: float = 0.7,
        semantic_weight: float = 1.0
) -> Tuple[MatchingResult, ...]:

    source_text_chunks = tuple(source_text_chunks)
    target_text_chunks = tuple(target_text_chunks)

    def get_positions(chunks: Iterable[str]) -> Iterable[int]:
        curr_pos = 0
        for chunk in chunks:
            yield curr_pos
            curr_pos += len(chunk)

    source_chunks_positions = tuple(get_positions(source_text_chunks))
    target_chunks_positions = tuple(get_positions(target_text_chunks))

    source_features = torch.stack(tuple(map(feature_extractor, source_text_chunks)))
    target_features = torch.stack(tuple(map(feature_extractor, target_text_chunks)))

    distance = pairwise_distances(source_features, target_features, metric='minkowski')
    semantic_confidence = torch.tensor(1 - (distance - distance.min()) / distance.max())

    total_source = len(source_text_chunks)
    total_target = len(target_text_chunks)

    matched_target: Set[int] = set()
    matched_pairs: List[MatchingResult] = []

    for source_idx, source in enumerate(source_text_chunks):
        if semantic_confidence[source_idx].max() < matching_confidence:
            # we cannot match any text chunk from target with enough confidence
            matched_pairs.append(MatchingResult(
                TextChunk(source_idx, source_chunks_positions[source_idx], source),
                None
            ))
            continue

        match_scores = []  # [0, 1]
        position_scores = []  # [0, 1]
        semantic_scores = []  # [0, 1]

        k = min(match_n_closest, max(total_target - source_idx, 0))
        if k == 0:
            matched_pairs.append(MatchingResult(
                TextChunk(source_idx, source_chunks_positions[source_idx], source),
                None
            ))
            continue

        # choose most confident matches
        match_idxes = torch.topk(semantic_confidence[source_idx], k=k).indices
        for possible_match_idx in match_idxes:
            match_scores.append(match(source, target_text_chunks[possible_match_idx]))
            position_scores.append(1 - abs(source_idx / total_source - possible_match_idx / total_target))
            semantic_scores.append(semantic_confidence[source_idx][possible_match_idx])

        full_score = sum([
            torch.tensor(match_scores) * matching_weight,
            torch.tensor(position_scores) * positional_weight,
            torch.tensor(semantic_scores) * semantic_weight
        ]) / (semantic_weight + matching_weight + positional_weight)

        chosen_score_idx = np.argmax(full_score)
        top_match_idx = match_idxes[chosen_score_idx].item()
        matched_target.add(top_match_idx)
        semantic_confidence[:, top_match_idx] = 0

        matched_pairs.append(MatchingResult(
            TextChunk(source_idx, source_chunks_positions[source_idx], source),
            TextChunk(top_match_idx, target_chunks_positions[source_idx], target_text_chunks[top_match_idx])
        ))

    res: List[MatchingResult] = []
    # restore target text chunks relative positions
    for idx in range(max(total_source, total_target)):
        if idx < total_source:
            res.append(matched_pairs[idx])

        if idx < total_target and idx not in matched_target:
            res.append(MatchingResult(None, TextChunk(idx, target_chunks_positions[idx], target_text_chunks[idx])))

    return tuple(res)


if __name__ == '__main__':
    src = nltk.tokenize.sent_tokenize('''В мире Москва занимает третье место, уступая лишь Нью-Йорку и Сан-Франциско.
Москва признана первой среди европейских городов в рейтинге инноваций,
помогающих в формировании устойчивости коронавирусу. Она опередила Лондон и
Барселону.
Среди мировых мегаполисов российская столица занимает третью строчку —
после Сан-Франциско и Нью-Йорка. Пятерку замыкают Бостон и Лондон. Рейтинг
составило международное исследовательское агентство StartupBlink.
Добиться высоких показателей Москве помогло почти 160 передовых решений,
которые применяются для борьбы с распространением коронавируса.
Среди них алгоритмы компьютерного зрения на основе искусственного
интеллекта. Это методика уже помогла рентгенологам проанализировать более трех
миллионов исследований.
Еще одно инновационное решение — облачная платформа, которая объединяет
пациентов, врачей, медицинские организации, страховые компании,
фармакологические производства и сайты.
Способствовали высоким результатам и технологии, которые помогают
адаптировать жизнь горожан во время пандемии. Это проекты в сфере умного туризма,
электронной коммерции и логистики, а также дистанционной работы и
онлайн-образования.
Эксперты агентства StartupBlink оценивали принятые в Москве меры с точки
зрения эпидемиологических показателей и влияния на экономику.''', language='russian')

    trgt = nltk.tokenize.sent_tokenize('''Бостон признан первым среди европейских городов в рейтинге инноваций,
помогающих в формировании устойчивости коронавирусу. Он опередил Лондон,
Барселону и Андроново.
В мире Бостон занимает третье место, уступая лишь Нью-Йорку и Сан-Франциско.
Андроново не участвовало в оценке в этом году. Рейтинг составило международное
исследовательское агентство StartupBlink.
Обойти преследователей Бостону помогло более 100 передовых решений,
которые применяются для борьбы с распространением коронавируса.
В свою очередь Андроново уже несколько лет не участвует в рейтинге по причине
отсутствия кислорода в атмосфере города и водорода в составе воды в реке Лене.
В качестве инновационного решения, позволяющего исправить положение,
неким человеком на улице было предложено использовать фаршированных
гонобобелем голубей для обеспечения регулярного авиасообщения с планетой
Железяка.
Другое предложенное решение оказалось ещё более странным, чем предыдущее
— облачная платформа, которая объединяет перистые и кучевые облака в
сверхмассивный кластер инновационных перисто-кучевых облаков.
Такого рода высокие технологии вряд ли помогут Андронову занять какое-либо
место в каком-нибудь конкурсе.''')

    trgt2 = nltk.tokenize.sent_tokenize('''В мире российская столица заняла третье место, обогнав Лондон и Барселону.
Москва заняла первое место среди европейских городов в рейтинге инноваций,
помогающих в борьбе с COVID-19, опередив Лондон и Барселону. Об этом сообщает
портал мэра и правительства Москвы. В мире российская столица заняла третье место,
уступив лишь Нью-Йорку и Сан-Франциско.
В российской столице применяются почти 160 передовых решений для борьбы с
распространением коронавируса. Среди них алгоритмы компьютерного зрения на
основе искусственного интеллекта, а такжеоблачная платформа, которая объединяет
пациентов, врачей, медицинские организации, страховые компании,
фармакологические производства и сайты. Способствовали высоким результатам и
технологии, которые помогают горожанам адаптироваться во время пандемии. Это
проекты в сфере умного туризма, электронной коммерции и логистики, а также
дистанционной работы и онлайн-образования.
Рейтинг составляется на базе глобальной карты инновационных решений по
борьбе с коронавирусом и оценивает около 100 ведущих городов и 40 стран мира.''')

    for matching_result in semantic_match(
            BertFeatureExtractor().extract_features,
            src,
            trgt2,
            match_n_closest=4,
            matching_confidence=0.4
    ):
        print(f'source: [{matching_result.source}]')
        print(f'target: [{matching_result.target}]\n')
