from feature_extraction.source_finder import *
import pandas as pd

def fake_detection(text):
    white_list = pd.read_pickle('white_list.pkl')
    sf = SourceFinder(white_list=white_list)
    article_info, distance = sf.find_source(text)
    return (article_info, distance)

text = 'До конца года партнерами благотворительных сервисов на портале Mos.ru могут быть 30 фондов, сообщила заместитель мэра Москвы Наталья Сергеевна в онлайн-встрече с представителями волонтерских организаций Московской области.\n\n"С помощью этого сервиса можно делать оплату всем зарегистрированным пользователям mos-ru в разделе ""Мои оплаты"", расположенном в разделе ""Мои оплаты""."\n\n«Сервис запущен в тестовом режиме 16 октября. Сейчас в нем зарегистрировано девять благотворительных проектов. В ближайшее время присоединятся еще 14. Планируется, что к концу года партнерами станут около 30 организаций», — отметила Наталья Сергунина.\n\nПо её словам, возможность платформы будет совершенствоваться в соответствии с предложениями благотворительных организаций и жителей Москвы, за первые несколько недель пользователи новых сервисов совершили более 10 тыс. переводов, средний размер оплаты составляет более 200 руб.\n\nДеньги могут быть направлены на помощь тяжело больным детям, людям с особенностями здоровья, малоимущим, а также на поддержку бездомных животных. К сервису уже подключились фонды «Справедливая помощь Доктора Лизы», «Со-единение», «Фонд продовольствия “Русь”», «Благо дари миру», «Ника», «Кораблик», «Женщины за жизнь», «Романсиада» и «Гольфстрим».\n\n"По словам исполнительного руководителя Фонда поддержки слепых ""Соединение"" Натальи Соковой, благодаря новым сервисам помощь для тех, кто нуждается в ней, станет доступнее."\n\n"""Фонды становятся ближе людям, для поддержки которых не нужно зайти на сайт организации и теперь можно получить благотворительные услуги между делом через портал Mos.ru, я уверен, что проект не только позволит увеличить число поступающих помощей, но и, главное, привлечет к процессу как можно большего количества людей"", - рассказала она."\n\nБлагодаря проекту благотворительные организации получат как дополнительную финансовую, так и информационную поддержку. «Такие сервисы помогают рассказать о работе благотворительных фондов, а это значит, что москвичи смогут больше узнать о нашей деятельности», — отметила Марина Зубова, президент фонда помощи тяжелобольным людям «Гольфстрим».\n\nПрисоединиться к проекту может любой фонд, который ведет социально значимую деятельность в столице более года и состоит в реестре благотворительных организаций Москвы. Чтобы максимально упростить процедуру подключения к сервису, был разработан специальный обучающий курс. Подробную информацию об участии, перечень документов и инструкцию по подключению можно найти на сайте «Душевная Москва».\n\nТворить добро: на Mos.ru появились благотворительные сервисыКак сделать Благотворительное Пожертвование на Mos.ru\n\nПравительство Москвы поддерживает благотворительные инициативы, организует мероприятия, выделяет гранты. В 2020 году общая сумма поддержки благотворительных фондов и организаций в столице превысила 300 миллионов рублей. Из них 67,6 миллиона рублей было выделено в рамках конкурса грантов Мэра Москвы для социально ориентированных некоммерческих организаций.'
art_inf, dist = fake_detection(text)
print(art_inf['url'].values)
print(art_inf['text'].values)
