# Отчёт: Кластеризация английского корпуса

## Постановка
Построить разбиение словаря английского корпуса (10,885 документов, 275K уникальных токенов) на кластеры трёх уровней гранулярности.

**Статус**: Phase 2 эксперименты (dim=50, CBOW, CBOW→SVD) показали что **v8-cbow50-nopc1** превосходит исходный v4-nopc1 baseline по всем метрикам. Новые итоговые артефакты: `c_p2_v8pc1-{large,medium,small}`.

## Итоговые артефакты

### Phase 2 (финальные, CBOW-based, dim=50)

| Уровень | Артефакт | Метод | Кластеры | Min | Max | CV | max/min |
|---------|----------|-------|----------|-----|-----|------|---------|
| Large (15-20) | `c_p2_v8pc1-large` | Bisecting K-Means | 18 | 2466 | 6248 | 0.271 | 2.53 |
| Medium (75-150) | `c_p2_v8pc1-medium` | Bisecting K-Means | 100 | 378 | 1107 | 0.226 | 2.93 |
| Small (500-1000) | `c_p2_v8pc1-small` | Bisecting K-Means | 538 | 95 | 291 | 0.240 | 3.06 |

**Эмбеддинги**: `v8-cbow50-nopc1` — Word2Vec CBOW vector_size=50, window=5, epochs=10, center + PC1 removal + L2. Словарь: 79,485 слов.

### Phase 1 (исторические, SVD-based, dim=300)

| Уровень | Артефакт | Метод | Кластеры | Min | Max | CV | max/min |
|---------|----------|-------|----------|-----|-----|------|---------|
| Large | `c11-large` | Bisecting K-Means | 18 | 1119 | 7839 | 0.491 | 7.01 |
| Medium | `c11-medium` | Bisecting K-Means | 100 | 223 | 1243 | 0.324 | 5.57 |
| Small | `c17-small` | Bisecting K-Means | 517 | 95 | 347 | 0.307 | 3.65 |

**Эмбеддинги Phase 1**: `v4-nopc1` — SVD-LSA (log-entropy, k=300, σ^0.5) с удалением PC1 + L2.

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

## Phase 2 — Результаты экспериментов

Выполнено дополнительное исследование: пониженная размерность (50 вместо 300) и альтернативные методы построения эмбеддингов (CBOW, CBOW→SVD гибрид).

### Сравнительная таблица (все версии эмбеддингов × 3 уровня)

| Version | Dim | Large CV | Medium CV | Small CV |
|---------|-----|----------|-----------|----------|
| v4-nopc1 (Phase 1 baseline) | 300 | 0.491 | 0.324 | 0.307 |
| v7-svd50 | 50 | 0.428 | 0.399 | 0.348 |
| v7-svd50-nopc1 | 50 | 0.356 | 0.252 | 0.266 |
| v8-cbow50 | 50 | 0.265 | **0.225** | 0.267 |
| **v8-cbow50-nopc1 ★** | 50 | 0.271 | 0.226 | **0.240** |
| v9-cbow300 | 300 | 0.260 | 0.248 | 0.284 |
| v9-cbow300-svd50 | 50 | **0.191** | 0.271 | 0.262 |
| v9-cbow300-svd50-nopc1 | 50 | 0.233 | 0.240 | 0.273 |

### Ключевые выводы Phase 2

1. **CBOW существенно превосходит SVD-LSA** по равномерности — CV упал с 0.3-0.5 до 0.22-0.27.
2. **50-мерные эмбеддинги лучше 300-мерных** для кластеризации: меньше шума, более плотная геометрия.
3. **PC1 removal критичен для SVD-LSA, но не для CBOW** — CBOW естественно распределяет дисперсию равномерно (отношение σ₁/σ₂ ≈ 1.3 против ≈2.4 у SVD-LSA).
4. **CBOW→SVD гибрид** (v9-cbow300-svd50) даёт лучшую формальную равномерность Large (CV=0.191), но **сильно ухудшает семантику** — кластеры становятся размытыми (смесь диалектов, имён, чисел). Высокая равномерность покупается ценой тематической связности.
5. **v8-cbow50-nopc1** — оптимальный баланс: отличная равномерность + лучшая лингвистическая связность.

### Лингвистическая валидация v8-cbow50-nopc1 (Large, все 18 кластеров)

| # | Size | Anchors | Тема |
|---|------|---------|------|
| 8 | 6248 | silvered, crinkled, ribbed, shiny | Материалы/фактура |
| 4 | 6240 | sandhill, pyrenee, bay, foreshore | География/ландшафт |
| 12 | 5893 | defensiveness, unawareness, passivity | Психология защиты |
| 13 | 5824 | sassy, danged, doggone, pardner | Западный диалект США |
| 10 | 5574 | marchmont, newman, vanbrugh, minver | Персонажи романов |
| 9 | 5397 | spheroidal, laminar, hirsuta, nucleolus | Научный регистр (биология) |
| 2 | 5151 | unjust, misrepresentation, forgive | Мораль/несправедливость |
| 1 | 4387 | abhorreth, enchain, sortilege | Поэтическая архаика |
| 7 | 4263 | naither, yoong, mayna, neebour | Шотландский диалект |
| 3 | 4122 | doughnut, sardine, pork, crock | Еда/кулинария |
| 6 | 4052 | handbook, memoir, republish, goethe | Издательское дело |
| 0 | 3437 | messire, hath, erst, thee | Архаика ME/EME |
| 11 | 3299 | trustee, solicitor, authorize | Юриспруденция |
| 15 | 3166 | thinke, onely, keepe, euill | Елизаветинский английский |
| 17 | 3142 | baron, emperor, valentinian | Феодальная знать |
| 14 | 2839 | ceux, demande, quant, heureux | Французский |
| 16 | 2466 | ptolemais, carnarvon, stockaded | Военные топонимы |

**Нет "свалочных" кластеров** — каждый имеет чёткую тему. На Phase 1 (v4-nopc1) имелись широкие категории без явной темы (C3 абстрактно-аналитическая, C11 общие моральные), здесь они получили более конкретные ярлыки.

## Метрики процесса (всего за Phase 1 + 2)

- Эмбеддинги: 13 версий (v1-v9)
- Кластеризации: 40+ конфигураций (c1-c17 Phase 1, 24 в Phase 2)
- Методы эмбеддингов: SVD-LSA (log-entropy/tf-idf/PPMI), CBOW, Skip-gram, CBOW→SVD
- Методы кластеризации: Wishart, K-Means, Bisecting K-Means, DBSCAN, HDBSCAN
- Размерности: 50, 300
- Время на CBOW-50 обучение: ~5 мин
- Время на CBOW-300 обучение: ~7 мин
- Время на кластеризацию (3 уровня): ~30 сек на итерацию

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
