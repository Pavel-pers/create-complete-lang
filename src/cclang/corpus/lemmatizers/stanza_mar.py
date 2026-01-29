import stanza
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from cclang.io.schemas import LemmaToken

class MahaNER:
    def __init__(self, model_name: str = "l3cube-pune/marathi-ner"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.eval()
        self.id2label = self.model.config.id2label
        self.propn_labels = {'Person', 'Location', 'Organization'}

    def predict_labels(self, tokens: list[str]) -> list[dict]:
        """
        Принимает список токенов, возвращает метки для каждого.
        """
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True
        )

        word_ids = inputs.word_ids()

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


class MarathiProcessor:
    def __init__(self):
        # Stanza для лемматизации
        stanza.download('mr', verbose=False)
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
            {'token': 'मुंबईत', 'lemma': 'मुंबई', 'pos': 'PROPN', 'ner': 'Location', 'score': 0.99, 'is_propn': True},
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
                               is_oov=False,
                               is_ambiguous=False
                               )
                )

        # 2. NER через MahaNER
        ner_results = self.ner.predict_labels(tokens)

        # 3. Объединяем
        results = []
        for i, item in enumerate(stanza_results):
            results.append(item)

        return results

    def process_batch(self, sentences: list[list[str]]) -> list[list[LemmaToken]]:
        """
        Батчевая обработка нескольких предложений.
        """
        # Stanza батч
        doc = self.nlp(sentences)

        all_results = []
        for sent_idx, sentence in enumerate(doc.sentences):
            tokens = sentences[sent_idx]

            # Stanza результаты
            stanza_results = [
                {'token': word.text, 'lemma': word.lemma, 'pos': word.upos}
                for word in sentence.words
            ]

            # NER результаты
            ner_results = self.ner.predict_labels(tokens)

            # Объединяем
            sent_results = []
            for i, item in enumerate(stanza_results):
                item['ner'] = ner_results[i]['label']
                item['ner_score'] = ner_results[i]['score']
                item['is_propn'] = ner_results[i]['label'] in self.ner.propn_labels
                sent_results.append(LemmaToken(
                    token=item['token'],
                    lemma=item['lemma'],
                    pos=item['pos'] or None,
                    analyses=[item['lemma']],
                    is_oov=False,
                    is_ambiguous=False
                ))

            all_results.append(sent_results)

        return all_results