# Отчёт: Кластеризация английского корпуса

## Постановка
Построить разбиение словаря английского корпуса (10,885 документов, 275K уникальных токенов) на кластеры трёх уровней гранулярности.

## Итоговые артефакты

| Уровень | Артефакт | Метод | Кластеры | Min | Max | CV | max/min |
|---------|----------|-------|----------|-----|-----|------|---------|
| Large (15-20) | `c11-large` | Bisecting K-Means | 18 | 1119 | 7839 | 0.49 | 7.01 |
| Medium (75-150) | `c11-medium` | Bisecting K-Means | 100 | 223 | 1243 | 0.32 | 5.57 |
| Small (500-1000) | `c17-small` | Bisecting K-Means | 517 | 95 | 347 | 0.31 | 3.65 |

**Эмбеддинги**: `v4-nopc1` — SVD-LSA (log-entropy, k=300, σ^0.5) с удалением первого главного компонента, L2-нормализация. Словарь: 79,485 слов (min_tf=5, min_df=3, без NER, стоп-слов, чисел).

## Ключевые решения

### 1. Wishart → Bisecting K-Means
Wishart (density-based) даёт крайне неравномерные кластеры на word embeddings из-за неравномерной плотности пространства. Один кластер (архаизмы/редкие слова) поглощает 40-50% словаря. Bisecting K-Means решает проблему, рекурсивно деля наибольший кластер.

### 2. Удаление первого SVD компонента
PC1 (σ₁=113, остальные ≤47) кодирует частотность слов, не семантику. Удаление PC1 из эмбеддингов v1 (79K→v4-nopc1) существенно улучшило равномерность кластеров:
- Large CV: 2.20 → 0.49
- Medium max/min: ∞ → 5.57

### 3. Фильтрация словаря
NER-плейсхолдеры (pron1, person1, ordinal1), стоп-слова, числа и однобуквенные токены отфильтрованы. Порог min_tf=5 оставляет достаточно слов для мелких кластеров.

## Примеры кластеров (medium, c11-medium)

| Кластер | Размер | Anchor-слова | Тема |
|---------|--------|-------------|------|
| 17 | 1234 | stew, sausage, cheese, biscuit, soup | Еда/кулинария |
| 48 | 1226 | mair, maun, ane, weel, muckle | Шотландский диалект |
| 59 | 1183 | blouse, jacket, collar, flannel, frock | Одежда |
| 15 | 1119 | schooner, anchor, mast, crew, hove | Мореплавание |
| 13 | 1120 | thoroughfare, pavement, street, passer, alley | Город/улицы |
| 76 | 1149 | client, transaction, executor, payment | Юриспруденция/финансы |
| 42 | 1181 | temperament, perception, egotism, emotional | Психология |

## Примеры кластеров (small, c17-small)

| Кластер | Размер | Anchor-слова | Тема |
|---------|--------|-------------|------|
| 40 | 347 | dog, terrier, yelp, puppy, barking | Собаки |
| 70 | 322 | calcareous, sedimentary, strata, stratum | Геология |
| 344 | 313 | que, dans, qu, une, avec | Французский |
| 333 | 299 | crimson, gorgeous, amber, tinted, hued | Цвета/оттенки |
| 450 | 295 | coon, mouse, frog, skunk, chipmunk | Животные |
| 329 | 296 | haue, selfe, euer, owne, loue | Елизаветинский английский |

## Метрики процесса

- Итераций: 17 конфигураций кластеризации (c1—c17)
- Версий эмбеддингов: 6 (v1—v6)
- Методы эмбеддингов: SVD + log-entropy, SVD + PPMI
- Методы кластеризации: Wishart, K-Means, Bisecting K-Means
- Время на эмбеддинги: ~14 мин (загрузка корпуса + TDM + SVD)
- Время на кластеризацию: ~10-40 сек на итерацию

## Структура артефактов

```
claude-experements/eng-corpus-wishart/data/
  embeddings/
    v4-nopc1/word_vectors.npy, vocab.tsv, meta.json
  clusters/
    c11-large/labels.npz, meta.json
    c11-medium/labels.npz, meta.json
    c17-small/labels.npz, meta.json
  validation/
    c11-large/, c11-medium/, c17-small/
      uniformity_report.json, linguistic_report.json, cluster_words.tsv
```
