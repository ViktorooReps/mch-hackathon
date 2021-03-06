from functools import partial
from pprint import pprint
from typing import Iterable, Callable, Optional, Tuple, NamedTuple, Dict, List

import nltk.tokenize
import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from fact_extraction.entity_extractor import EntityExtractor
from fact_extraction.helper import get_fact_consistency
from fact_extraction.model import Entity, EntityType
from feature_extraction.bert import BertFeatureExtractor
from feature_extraction.sequence_matcher.semantic import MatchingResult, semantic_match, TextChunk


class ScoredMatchingResult(NamedTuple):
    source: Optional[TextChunk]  # кусок текста статьи первоисточника
    target: Optional[TextChunk]  # соответствующий кусок статьи, подозревающейся на фейковость
    matched: bool  # было ли совпадение кусков

    # are present only if matched is True
    fact_score: Optional[float]  # оценка правдивости упомянутых фактов
    is_fake: Optional[bool]  # вердикт о фейковости


class OriginComparisonResults(NamedTuple):
    matches: Tuple[ScoredMatchingResult, ...]  # совпадения отсортированные в порядке упомянания в текстах
    matched_proportion: float  # доля кусков текста, для которых было найдено совпадение
    features: Dict[str, float]  # выделенные признаки


# FIXME: possible bottleneck
def get_chunk_entities(chunk: Optional[TextChunk], entities: Iterable[Entity]) -> Iterable[Entity]:
    if chunk is None:
        return tuple()

    chunk_start = chunk.text_position
    chunk_end = chunk_start + len(chunk.text)
    for entity in sorted(entities, key=lambda e: e.start):
        if entity.start >= chunk.text_position and entity.end < chunk_end:
            yield entity
        if entity.end >= chunk_end:
            break


def get_entity_count(
        entity_type: EntityType,
        chunk_entities: Iterable[Tuple[Entity, ...]]
) -> NDArray:

    def type_filter(entity: Entity) -> bool:
        return entity.label == entity_type

    result = []
    for entities in chunk_entities:
        result.append(len(tuple(filter(type_filter, entities))))
    return np.array(result)


def get_unmatched_entities(
        entity_type: EntityType,
        source_chunk_entities: Iterable[Tuple[Entity, ...]],
        target_chunk_entities: Iterable[Tuple[Entity, ...]]
) -> NDArray:

    result = []
    for source_entities, target_entities in zip(source_chunk_entities, target_chunk_entities):
        source_entity_strs = {entity.text for entity in source_entities if entity.label == entity_type}
        target_entity_strs = {entity.text for entity in target_entities if entity.label == entity_type}

        if not len(source_entity_strs):
            result.append(0.0)
            continue

        unmatched = source_entity_strs.difference(target_entity_strs)
        result.append(len(unmatched) / len(source_entity_strs))

    return np.array(result)


def get_match_for_entity_type(
        entity_type: EntityType,
        origin_chunk_entities: Iterable[Tuple[Entity, ...]],
        article_chunk_entities: Iterable[Tuple[Entity, ...]]
) -> NDArray:

    def type_filter(entity: Entity) -> bool:
        return entity.label == entity_type

    result = []
    for origin_entities, article_entities in zip(origin_chunk_entities, article_chunk_entities):
        result.append(get_fact_consistency(
            true_facts=filter(type_filter, origin_entities),
            target_facts=filter(type_filter, article_entities)
        ))

    return np.array(result)


class ArticleOriginFeatureExtractor:

    """Класс для извлечения признаков для обучения ML модели из пар (текст-первоисточник, текст)"""

    def __init__(
            self,
            text_chunker: Callable[[str], Iterable[str]],
            entity_extractor: EntityExtractor,
            feature_extractor: Callable[[str], Tensor],
            *,
            fake_score_threshold: float = 0.5
    ):
        """

        :param text_chunker: - функция, разделяющая текст на куски для последующего соответствия
        :param entity_extractor: - извлекатель сущностей
        :param feature_extractor: - извлекатель семантических признаков
        :param fake_score_threshold: - максимальное значение оценки несовпадения фактов, чтобы считать, что в данном куске текста
            присутствует фейк (не влияет на извлеченные признаки)
        """
        self._chunker = text_chunker
        self._entity_extractor = entity_extractor
        self._feature_extractor = feature_extractor
        self._fake_score_threshold = fake_score_threshold

    def extract_features(self, article_text: str, possible_origins: Iterable[str]) -> OriginComparisonResults:
        """ Извлечь признаки и сопоставление кусков текста из текста статьи и возможных первоисточников

        :param article_text: - текст статьи
        :param possible_origins: - текст возможных первоисточников
        :return: OriginComparisonResults
        """
        possible_origins = tuple(possible_origins)
        article_chunks = self._chunker(article_text)

        origin_matching_result: Optional[Tuple[MatchingResult, ...]] = None
        origin_text: Optional[str] = None
        origin_matched_proportion = 0.0
        for po_text, po_chunks in zip(possible_origins, map(self._chunker, possible_origins)):
            matching_result = semantic_match(self._feature_extractor, po_chunks, article_chunks)
            matched_proportion = self._matched_proportion(matching_result)

            print(f'Article text: {article_text[:100]}')
            print(f'Origin text: {po_text[:100]}')
            print(f'Matched: {matched_proportion}\n')

            if matched_proportion >= origin_matched_proportion:
                origin_matched_proportion = matched_proportion
                origin_matching_result = matching_result
                origin_text = po_text

        def filter_matched(match: MatchingResult) -> bool:
            return match.source is not None and match.target is not None

        if origin_matching_result is None:
            print('No origin')
            raise ValueError('No origin.')

        origin_entities = self._entity_extractor.get_entities(origin_text)
        article_entities = self._entity_extractor.get_entities(article_text)

        fact_scores = self._get_fact_scores_for_matches(origin_entities, article_entities, filter(filter_matched, origin_matching_result))

        fact_score_iterator = iter(fact_scores)
        scored_matches: List[ScoredMatchingResult] = []
        for origin_match in origin_matching_result:
            fact_score = None
            if filter_matched(origin_match):
                fact_score = next(fact_score_iterator)

            scored_matches.append(ScoredMatchingResult(
                source=origin_match.source,
                target=origin_match.target,
                matched=(fact_score is not None),
                fact_score=fact_score,
                is_fake=None if fact_score is None else fact_score < self._fake_score_threshold
            ))

        result = OriginComparisonResults(matches=tuple(scored_matches), matched_proportion=origin_matched_proportion, features={})
        self._fill_features(result, origin_entities, article_entities)
        return result

    @staticmethod
    def _fill_features(results: OriginComparisonResults, origin_entities: Iterable[Entity], article_entities: Iterable[Entity]):
        chunk_features: Dict[str, NDArray] = {}

        origin_entities = tuple(origin_entities)
        article_entities = tuple(article_entities)

        origin_chunks: List[TextChunk] = [match.source for match in results.matches]
        article_chunks: List[TextChunk] = [match.target for match in results.matches]

        origin_chunk_entities = tuple(map(tuple, map(partial(get_chunk_entities, entities=origin_entities), origin_chunks)))
        article_chunk_entities = tuple(map(tuple, map(partial(get_chunk_entities, entities=article_entities), article_chunks)))

        for entity_type in EntityType:
            entity_type_name = entity_type.value

            feat_name = f'{entity_type_name}_ent_origin_count'
            src_count = get_entity_count(entity_type, origin_chunk_entities)
            chunk_features[feat_name] = src_count

            feat_name = f'{entity_type_name}_ent_article_count'
            art_count = get_entity_count(entity_type, article_chunk_entities)
            chunk_features[feat_name] = art_count

            feat_name = f'{entity_type_name}_ent_count_diff'
            chunk_features[feat_name] = src_count - art_count

            feat_name = f'{entity_type_name}_unmatched_origin_entities'
            chunk_features[feat_name] = get_unmatched_entities(entity_type, origin_chunk_entities, article_chunk_entities)

            feat_name = f'{entity_type_name}_unmatched_article_entities'
            chunk_features[feat_name] = get_unmatched_entities(entity_type, article_chunk_entities, origin_chunk_entities)

            feat_name = f'{entity_type_name}_avg_match'
            chunk_features[feat_name] = get_match_for_entity_type(entity_type, article_chunk_entities, origin_chunk_entities)

        for aggregate_function, function_name in ((np.mean, 'mean'), (np.max, 'max'), (np.min, 'min')):
            for feat_name, feat_values in chunk_features.items():
                results.features[feat_name + '_' + function_name] = aggregate_function(feat_values)

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

    @staticmethod
    def _get_fact_scores_for_matches(
            origin_entities: Iterable[Entity],
            article_entities: Iterable[Entity],
            matches: Iterable[MatchingResult]
    ) -> Iterable[float]:

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
    for m in res.matches:
        if m.source is not None and m.target is not None:
            print(f'Match!\n  fact_score: {m.fact_score}\n  ')
        if m.source is not None:
            print(f'Origin: {m.source.text}')
        if m.target is not None:
            print(f'Article: {m.target.text}')
        print('-' * 20)
    pprint(res.features)
