import threading

import stanza
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from cclang.io.schemas import LemmaToken

_THREAD_LOCAL = threading.local()


class MahaNER:
    def __init__(self, model_name: str = "l3cube-pune/marathi-ner", offline: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=offline)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, local_files_only=offline)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label
        self.propn_labels = {'Person', 'Location', 'Organization'}

    def predict_labels(self, tokens: list[str]) -> list[dict]:
        """
        Принимает список токенов, возвращает метки для каждого.
        """
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        word_ids = encoding.word_ids()
        inputs = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=-1)[0]
        scores = torch.softmax(outputs.logits, dim=-1)[0]

        results = []
        prev_word_id = None

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id == prev_word_id:
                continue

            label = self.id2label[predictions[idx].item()]
            clean_label = label.split('-')[-1] if '-' in label else label

            results.append({
                'token': tokens[word_id],
                'label': clean_label,
                'bio_label': label,
                'score': scores[idx][predictions[idx]].item()
            })

            prev_word_id = word_id

        return results


class MarathiStanzaProcessor:
    def __init__(self):
        # Stanza для лемматизации (модель должна быть уже скачана)
        self.nlp = stanza.Pipeline(
            'mr',
            processors='tokenize,pos,lemma',
            tokenize_pretokenized=True,
            verbose=False
        )

        self.ner = MahaNER()

    def process(self, tokens: list[str]) -> list[LemmaToken]:
        """
        Возвращает для каждого токена: лемму, POS, NER-метку.

        Input:  ['मुंबईत', 'राहुलने', 'भाषण', 'केले']
        Output: [
            LemmaToken(token='मुंबईत', lemma='मुंबई', pos='PROPN', ner_result='Location', ...),
            ...
        ]
        """
        # 1. Лемматизация через Stanza
        doc = self.nlp([tokens])
        stanza_results: list[LemmaToken] = []
        for sentence in doc.sentences:
            for word in sentence.words:
                stanza_results.append(
                    LemmaToken(token=word.text,
                               lemma=word.lemma,
                               pos=word.upos,
                               analyses=[word.lemma],
                               )
                )

        # 2. NER через MahaNER
        ner_results = self.ner.predict_labels(tokens)

        # 3. Объединяем NER-метки
        results = []
        for i, item in enumerate(stanza_results):
            if i < len(ner_results):
                ner_label = ner_results[i]['label']
                is_ne = ner_label != "Other"
                item.is_ne = is_ne
                item.ner_result = ner_label if is_ne else None
            results.append(item)

        return results

    def process_batch(self, sentences: list[list[str]]) -> list[list[LemmaToken]]:
        """
        Батчевая обработка нескольких предложений.
        Оптимизировано: NER батчится по чанкам до 100 токенов (лимит BERT 512).
        """
        MAX_NER_TOKENS = 80  # Conservative: BERT subword tokenization can expand 3-6x

        # 1. Stanza батч - лемматизация и POS
        doc = self.nlp(sentences)

        # 2. Извлекаем токены из Stanza результатов
        sentence_tokens = []
        for sentence in doc.sentences:
            sentence_tokens.append([word.text for word in sentence.words])

        # 3. Группируем предложения в NER-чанки по лимиту токенов
        ner_chunks = []  # list of (sentence_indices, flat_tokens)
        current_chunk_indices = []
        current_chunk_tokens = []

        for sent_idx, tokens in enumerate(sentence_tokens):
            if len(current_chunk_tokens) + len(tokens) > MAX_NER_TOKENS and current_chunk_tokens:
                # Сохраняем текущий чанк и начинаем новый
                ner_chunks.append((current_chunk_indices.copy(), current_chunk_tokens.copy()))
                current_chunk_indices = []
                current_chunk_tokens = []

            current_chunk_indices.append(sent_idx)
            current_chunk_tokens.extend(tokens)

        if current_chunk_tokens:
            ner_chunks.append((current_chunk_indices, current_chunk_tokens))

        # 4. Обрабатываем NER по чанкам
        sentence_ner_results = [None] * len(doc.sentences)

        for chunk_indices, chunk_tokens in ner_chunks:
            if not chunk_tokens:
                continue

            ner_results = self.ner.predict_labels(chunk_tokens)

            # Распределяем результаты по предложениям
            offset = 0
            for sent_idx in chunk_indices:
                sent_len = len(sentence_tokens[sent_idx])
                sentence_ner_results[sent_idx] = ner_results[offset:offset + sent_len]
                offset += sent_len

        # 5. Собираем финальные результаты
        all_results = []
        for sent_idx, sentence in enumerate(doc.sentences):
            ner_slice = sentence_ner_results[sent_idx] or []

            sent_results = []
            for word_idx, word in enumerate(sentence.words):
                ner_label = ner_slice[word_idx]['label'] if word_idx < len(ner_slice) else 'Other'
                is_ne = ner_label != 'Other'
                sent_results.append(LemmaToken(
                    token=word.text,
                    lemma=word.lemma,
                    pos=word.upos or None,
                    analyses=[word.lemma],
                    is_ne=is_ne,
                    ner_result=ner_label if is_ne else None
                ))

            all_results.append(sent_results)

        return all_results


def batch_stanza_lemmatize(sentences: list[list[str]]) -> list[list[LemmaToken]]:
    return get_thread_local_stanza_preprocessor().process_batch(sentences)


def get_thread_local_stanza_preprocessor() -> MarathiStanzaProcessor:
    """
    Create one Analyzer per thread to avoid shared-state issues and reduce init overhead.
    """
    if not hasattr(_THREAD_LOCAL, "stanza_processor"):
        _THREAD_LOCAL.stanza_processor = MarathiStanzaProcessor()
    return _THREAD_LOCAL.stanza_processor
