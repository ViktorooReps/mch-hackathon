from typing import Iterable, Callable, Optional, Tuple, NamedTuple

import nltk.tokenize
from torch import Tensor

from fact_extraction.entity_extractor import EntityExtractor
from fact_extraction.helper import get_fact_consistency
from fact_extraction.model import Entity
from feature_extraction.bert import BertFeatureExtractor
from feature_extraction.sequence_matcher.semantic import MatchingResult, semantic_match, TextChunk


class OriginComparisonFeatures(NamedTuple):
    matches: Tuple[MatchingResult, ...]
    fact_scores: Tuple[float, ...]
    matched_proportion: float


class ArticleOriginFeatureExtractor:

    def __init__(
            self,
            text_chunker: Callable[[str], Iterable[str]],
            entity_extractor: EntityExtractor,
            feature_extractor: Callable[[str], Tensor]
    ):
        self._chunker = text_chunker
        self._entity_extractor = entity_extractor
        self._feature_extractor = feature_extractor

    def extract_features(self, article_text: str, possible_origins: Iterable[str]) -> OriginComparisonFeatures:
        possible_origins = tuple(possible_origins)
        article_chunks = self._chunker(article_text)

        origin_matching_result: Optional[Tuple[MatchingResult, ...]] = None
        origin_text: Optional[str] = None
        origin_matched_proportion = 0.0
        for po_text, po_chunks in zip(possible_origins, map(self._chunker, possible_origins)):
            matching_result = semantic_match(self._feature_extractor, po_chunks, article_chunks)
            matched_proportion = self._matched_proportion(matching_result)

            if matched_proportion >= origin_matched_proportion:
                origin_matched_proportion = matched_proportion
                origin_matching_result = matching_result
                origin_text = po_text

        def filter_matched(match: MatchingResult) -> bool:
            return match.source is not None and match.target is not None

        if origin_matching_result is None:
            raise ValueError('No origin.')

        fact_scores = self._get_fact_scores_for_matches(origin_text, article_text, filter(filter_matched, origin_matching_result))
        return OriginComparisonFeatures(origin_matching_result, tuple(fact_scores), origin_matched_proportion)

    @staticmethod
    def _matched_proportion(matched: Iterable[MatchingResult]) -> float:
        total_chunks = 0
        matched_chunks = 0
        for match in matched:
            total_chunks += 1  # there is always at least one non-None attribute

            if match.source is not None and match.target is not None:
                total_chunks += 1
                matched_chunks += 2

        return matched_chunks / total_chunks

    def _get_fact_scores_for_matches(
            self,
            origin_text: str,
            article_text: str,
            matches: Iterable[MatchingResult]
    ) -> Iterable[float]:

        origin_entities = self._entity_extractor.get_entities(origin_text)
        article_entities = self._entity_extractor.get_entities(article_text)

        # FIXME: possible bottleneck
        def get_chunk_entities(chunk: TextChunk, entities: Iterable[Entity]) -> Iterable[Entity]:
            chunk_start = chunk.text_position
            chunk_end = chunk_start + len(chunk.text)
            for entity in entities:
                if entity.start >= chunk.text_position and entity.end < chunk_end:
                    yield entity

        for match in matches:
            yield get_fact_consistency(
                true_facts=get_chunk_entities(match.source, origin_entities),
                target_facts=get_chunk_entities(match.target, article_entities)
            )


if __name__ == '__main__':
    aofe = ArticleOriginFeatureExtractor(
        text_chunker=nltk.tokenize.sent_tokenize,
        entity_extractor=EntityExtractor(),
        feature_extractor=BertFeatureExtractor().extract_features
    )

    art = '''Бостон признан первым среди европейских городов в рейтинге инноваций,
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
место в каком-нибудь конкурсе.'''

    po = ['''В мире Москва занимает третье место, уступая лишь Нью-Йорку и Сан-Франциско.
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
зрения эпидемиологических показателей и влияния на экономику.''']

    res = aofe.extract_features(art, po)
    print(f'Matched proportion: {res.matched_proportion}\n')
    print('-' * 20)
    fact_score_iter = iter(res.fact_scores)
    for match in res.matches:
        if match.source is not None and match.target is not None:
            print(f'Match!\n  fact_score: {next(fact_score_iter)}\n  ')
        if match.source is not None:
            print(f'Origin: {match.source.text}')
        if match.target is not None:
            print(f'Article: {match.target.text}')
        print('-' * 20)
