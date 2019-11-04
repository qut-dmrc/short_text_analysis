import pandas as pd
from bert import tokenization

from bert_train import convert_dataframe_to_features, replace_special_tokens

VOCAB_FILE = 'gs://platform_governance_analysis/bert/BERT_uncased_L-24_H-1024_A-16_vocab-NS.txt'


class TestPreProcess(object):
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        """This method is run once for each class before any tests are run"""
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_preprocess(self):
        test_dataset = [
            {'guid': '0',
             'text_a': 'All I can hear from each table “ you fucking niggers can’t touch me. “Fuck you nigger” “ya mama is sucking my dick bitch ass nigga” if you know you know',
             'label': None},
            {'guid': '0', 'text_a': '"👹:You monster fucker--"', 'label': None},
            {'guid': '0',
             'text_a': '"RT @honeydollheart: him: hi u stupid cunt haha what up me: https://t.co/lCMsUcSlzP"',
             'label': None},
            {'guid': '0', 'text_a': 'SAMPLETEXT', 'label': None},
        ]
        df = pd.DataFrame(test_dataset)

        list_guids_and_features = convert_dataframe_to_features(df, vocab_file=VOCAB_FILE, do_lower_case=False,
                                                                label_list=[None], max_seq_length=256,
                                                                is_predicting=True)

        print(list_guids_and_features)

    def test_special_replacements(self):
        tokenizer = tokenization.FullTokenizer(
            vocab_file=VOCAB_FILE, do_lower_case=True)
        input_texts = [
            'All I can hear from each table “ you fucking niggers can’t touch me. “Fuck you nigger” “ya mama is sucking my dick bitch ass nigga” if you know you know',
            '"👹:You monster fucker--"',
            '"RT @honeydollheart: him: hi u stupid cunt haha what up me: https://t.co/lCMsUcSlzP"',
            'SAMPLETEXT',
        ]
        output_texts = []

        for text in input_texts:
            text = tokenization.convert_to_unicode(text)
            text = replace_special_tokens(text)
            text = tokenizer.tokenize(text)
            output_texts.append(text)

        print(output_texts)
