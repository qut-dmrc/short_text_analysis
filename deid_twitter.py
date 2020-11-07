#
# Remove identifying information from tweet datasets and (as far as possible) tweets.
# All IDs will be hashed with HMAC256 using a secret key in config.py.
import base64
import csv
import hmac
import re
from hashlib import sha256

import pandas as pd
from docopt import docopt
import logging
import os
from config import HMAC_KEY

logger = logging.getLogger(__name__)


def main():
    """Remove identifying information from tweet datasets and (as far as possible) tweets.
    All IDs will be hashed with HMAC256 using a secret key in config.py.

    Usage:
      deid_twitter.py [-v] <input_file> [<output_file>]

    Options:
      -h --help                 Show this screen.
      -v --verbose              Increase verbosity for debugging.

      --version  Show version.

      <input_file>                CSV, JSON, or XLSX file to read and deidentify
      [<output_file>]             If not present, will add '.deid' suffix

    """

    args = docopt(main.__doc__, version='Deidentify Twitter 0.1')

    if args['<input_file>'][-4:] == '.csv':
        df = pd.read_csv(args['<input_file>'])
    elif args['<input_file>'][-4:] == '.xls' or args['<input_file>'][-5:] == '.xlsx':
        df = pd.read_excel(args['<input_file>'])
    elif args['<input_file>'][-5:] == '.json':
        df = pd.read_json(args['<input_file>'], orient='records')
    else:
        raise IOError(f"No filename passed: {args['<input_file>']}")

    df = deid_df(df)

    if args['<output_file>']:
        out_name = args['<output_file>']
    else:
        out_name = os.path.splitext(args['<input_file>'])
        out_name = out_name[0] + '.deid' + out_name[1]
        out_name = ''.join(out_name)

    final_export_name = out_name
    if os.path.exists(out_name):
        logger.error("Output file exists! Adding suffix.")
        final_export_name = out_name
        i=0
        while os.path.exists(final_export_name):
            final_export_name = out_name + f'.{i}'

    if out_name[-4:] == '.csv':
        df.to_csv(final_export_name, quoting=csv.QUOTE_ALL)
    elif out_name[-4:] == '.xls' or out_name[-5:] == '.xlsx':
        df.to_excel(final_export_name)
    elif out_name[-5:] == '.json':
        df.to_json(final_export_name, orient='records')

def deid_df(df):
    key = HMAC_KEY.encode()
    for col in df.columns:
        if col == 'id' or col[-3:] == '_id':
            logger.info(f'Hashing {col}.')
            df.loc[:, col] = df[col].apply(lambda x: hmac_sha256(x, key))
        elif col == 'text' or col[-3:] == '_text':
            logger.info(f'Replacing text in {col}.')
            df.loc[:, col] = replace_text_col(df[col])
    return df


def hmac_sha256(identifier, key):
    """ Convert an identifier to a pseudonymous hash.
    We use HMAC with SHA256 to hash identifiers. This allows us to retain referential integrity without
    storing personally identifiable information. We use a secret key to avoid dictionary attacks.
    """

    if not identifier:
        return None

    if isinstance(identifier, bytes):
        pass
    elif isinstance(identifier, str):
        identifier = identifier.encode()
    else:
        identifier = str(identifier).encode()

    h = hmac.new(key, identifier, sha256)
    encoded_id = base64.b64encode(h.digest()).decode()
    return encoded_id

def replace_text_col(col):

    # do retweets first so that the username is still there
    pattern = re.compile(r'(\"@|\brt @|\bmt @|\bvia @)(?=@\w+)', flags=re.IGNORECASE | re.MULTILINE)
    replace = "xxretweet "
    col = col.replace(to_replace=pattern, value=replace, regex=True)

    pattern = re.compile(r"\B#\b")
    replace = "xxhashtagxx "
    col = col.replace(to_replace=pattern, value=replace, regex=True)

    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', flags=re.IGNORECASE | re.MULTILINE)
    replace = "xxurlxx "
    col = col.replace(to_replace=pattern, value=replace, regex=True)

    pattern = re.compile(r"\B@\w{1,15}\b", flags=re.IGNORECASE | re.MULTILINE)
    replace = "xxusernamexx "
    col = col.replace(to_replace=pattern, value=replace, regex=True)

    return col

if __name__ == '__main__':
    main()