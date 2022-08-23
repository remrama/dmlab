"""Text analysis module.
"""
from collections import Counter
import csv
from pathlib import Path
import random
import re

import contractions
import liwc
import nltk
import pandas as pd
import spacy
import unidecode

from .io import *
from .plotting import *


__all__ = [
    
    "load_spacy_model",
    "normalize",
    "count_words",

    "segment_fixed_size",
    "segment_fixed_count",

    "load_liwc_dictionary",
    "build_liwc_dictionary",
    "liwc_tokenize",
    "liwc_single_doc",
    
    "plot_timecourse",

]



########################################################
# Utilities
########################################################

def load_spacy_model(model_name="en_core_web_sm"):
    """Load spacy model and download it if necessary.

    Parameters
    ----------
    model_name : str
        Name of spacy model to load. See https://spacy.io/models

    Returns
    -------
    nlp : spacy model

    Notes
    -----
    Check for spacy model already installed and downloads
    it if not already there. Also warns about certain models.
    I like to use en_core_web_lg or en_core_web_trf.
    """

    if model_name.endswith("sm"):
        print("Warning, small models are bad at entity recognition! Use only for testing.")

    if not spacy.util.is_package(model_name):
        resp = input(f"{model_name} not found -- download now?? (Y/n)")
        if resp.lower() in ["", "y"]:
            spacy.cli.download(model_name)

    # For the lg model you can disable most other thing and this
    # will speed up the spaCy/nlp stuff. You can't get lemmas but okay.
    # SPACY_PIPE_DISABLES = ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]
    # nlp = spacy.load(SPACY_MODEL, disable=SPACY_PIPE_DISABLES)
    nlp = spacy.load(model_name)
    nlp.add_pipe("merge_entities") # so "john paul" gets treated as a single entity
    # nlp.add_pipe("spacytextblob")
    return nlp



########################################################
# Preprocessing
########################################################

# nlp = load_spacy_model("en_core_web_sm")

def normalize(s, extra_clean=True, min_characters=30):
    """Normalize text. Return a string that is ASCII and printable.
    
    Parameters
    ----------
    s : str
        Text string to be converted.
    extra_clean : bool
        If True, perform additional steps like replacing contractions.
    """
    # Replace annoying unicode surrogates (??) that cause warnings in unidecode.
    surrogates = r"[\ud83d\ud83c\udf37\udf38\udf39\udf3a\udc2c]+"
    s = re.sub(surrogates, " ", s)
    # Unidecode does the heavy-lifting on conversion to ASCII.
    s = unidecode.unidecode(s, errors="ignore", replace_str="")
    # # Replace hex/control characters.
    # s = re.sub(r"[^\x00-\x7F]+", r"", s).strip()
    # # # Replace some non-printable whitespace characters that are technically ASCII but not printable (and unidecode doesn't catch).
    # # text = re.sub(r"[\x1b\x7f]+", " ", text)
    # # Replace all hex codes
    # s = re.sub(r"\[^x[0-9a-f]{2}\]", "", s)
    # # Reduce to single whitespaces.
    # s = re.sub(r"\s+", " ", s)
    # # Replace the few stupid apostrophe representations.
    # s = s.replace("&#39;", "'") # unidecode does this.
    # assert s.isascii() and s.isprintable(), f"Text is not ASCII and printable:\n{s}"
    if not s.isascii() and s.isprintable():
        print("fix this bullshit!!!!")
    # could force unicode with s.decode("utf8").encode("ascii", errors="ignore")
    # or import string; string.printable
    if extra_clean:
        # Replace ampersands.
        s = s.replace("&", "and")
        # Replace contractions with full words.
        s = contractions.fix(s, slang=True)
        # Replace any sequence of 4+ characters with 1 of that character.
        # Gets rid of stuff like whoaaaaaaaaaaaa and --------------------
        # Will lead to some errors because it replaces with 1 letter but sometimes will need 2.
        s = re.sub(r"(.)\1{3,}", r"\1", s, flags=re.IGNORECASE)
        # Replace double-quotes with single-quotes (to avoid file parsing errors).
        s = s.replace('"', "'")
    if s and len(s) > min_characters:
        return s

def count_words(s):
    """Count number of words in a given string."""
    wc = sum([ re.search(r"[a-zA-Z]", tok) is not None for tok in s.split() ])
    return wc

def preprocess_spacy(
        s,
        shuffle=False,
        pos_to_remove=None,
        entities_to_redact=None,
    ):
    """takes a spaCy doc.
    Not stressing too hard on this because it's likely
    that one will want to tokenize/lemmatize their own way.
    Also some thing are easy to do later and not always wanted (like removing stop words).
    # Select which of the following named entity labels should get removed/replaced.
    """
    available_entities = [
        "CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE",
        "LAW", "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT",
        "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART",
    ]
    # See full list of spaCy POS tags: https://github.com/explosion/spaCy/blob/2d89dd9db898e66058bf965e1b483b0019ce1b35/spacy/glossary.py#L22
    available_pos = [
        # Universal
        "ADJ", # adjective
        "ADP", # adposition
        "ADV", # adverb
        "AUX", # auxiliary
        "CCONJ", # coordinating conjunction
        "DET", # determiner
        "INTJ", # interjection
        "NOUN", # noun
        "NUM", # numeral
        "PART", # particle
        "PRON", # pronoun
        "PROPN", # proper noun
        "PUNCT", # punctuation
        "SCONJ", # subordinating conjunction
        "SYM", # symbol
        "VERB", # verb
        "X", # other
    ]
    if entities_to_redact is None:
        entities_to_redact = []
    assert isinstance(entities_to_redact, list)
    assert all([ x in available_entities for x in entities_to_redact ])
    if pos_to_remove is None:
        pos_to_remove = []
    assert isinstance(pos_to_remove, list)
    assert all([ x in available_pos for x in pos_to_remove ])

    doc = nlp(s)

    tokens = []
    lemmas = []
    redactions = []
    for t in doc:
        if t.like_email:
            tokens.append("[[EMAIL]]")
            redactions.append(t)
        elif t.like_url:
            tokens.append("[[URL]]")
            redactions.append(t)
        elif t.ent_type_ in entities_to_redact:
            tokens.append(f"[[{t.ent_type_}]]")
            redactions.append(t)
        elif (t.is_alpha #) and (t.is_ascii
            ) and (len(t) >= 2
            ) and (not t.is_stop
            ) and (not t.like_num # is_digit
            # ) and (not t.is_oov
            ) and (not t.pos_ in pos_to_remove
            ):
            tokens.append(t)
            lemmas.append(t.lemma_.lower())

    if shuffle:
        random.shuffle(lemmas)

    # Get counts from raw/normalized input.
    n_chars = len(s)
    n_tokens = len(doc)
    n_words = sum( t.is_alpha for t in doc )
    n_lemmas = len(lemmas)

    if lemmas:
        return {
            "text": s,
            "tokens": tokens,
            "lemmas": lemmas,
            "n_characters": n_chars,
            "n_tokens": n_tokens,
            "n_words": n_words,
            "n_lemmas": n_lemmas,
        }



##### some general splitting functions pulled from stack overflow

def segment_fixed_size(lst, n):
    """Yield successive n-sized chunks/segments from lst.
    https://stackoverflow.com/a/312464
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def segment_fixed_count(l, n):
    """Yield n number of sequential chunks/segments from l.
    https://stackoverflow.com/a/54802737
    - See also np.array_split(l, n)
    """
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield l[si:si+(d+1 if i < r else d)]


########################################################
# LIWC
########################################################

def build_liwc_dictionary():
    # See lucidemo code, among others.
    raise NotImplementedError


def load_liwc_dictionary(dictionary):
    """
    Parameters
    ----------
    dictionary : str
        name for loading or filepath
    """
    available_dicts = [
        "LIWC2015",
        "AgencyCommunion",
        "ArcOfNarrative",
    ]

    if dictionary not in available_dicts:
        dict_path = Path(dictionary)
        if dict_path.suffix != ".dic" or not dict_path.is_file():
            raise ValueError("Dictionary must be the name of builtin or a filepath to existing .dic file.")
    else:
        dict_path = Path(f"./mylocaldir/{dictionary}")
    parse, category_names = liwc.load_token_parser(dict_path)
    lexicon, category_names_ = liwc.read_dic(dict_path) # { word: category_list }
    return parse, category_names, lexicon


LiwcTokenizer = nltk.tokenize.TweetTokenizer()
def liwc_tokenize(s, min_tokens=5):
    """Turns string into list of tokens if more than min_tokens.

    LIWC vocab includes lots of apostrophed and hyphenated words, and emojis.
    The nltk tweet tokenizer is good for this situation, but I also wanna get rid of punctuation.
    
    This could just spit out a list/generator
    of token segments that could be iterated over.
    This would make the actual looping code at the
    bottom more concise.
    """
    # Break into tokens.
    tokens = LiwcTokenizer.tokenize(s)
    # Remove isolated puncuation and lowercase.
    tokens = [ t.lower() for t in tokens if not (len(t)==1 and not t.isalpha()) ]
    if len(tokens) >= min_tokens:
        return tokens

def liwc_single_doc(tokens, parser, categories, n_decimals=4):
    """Turn list of tokens into category frequencies.
    
    Parameters
    ----------
    tokens : list
        List of tokens.
    categories : list
        List of LIWC categories.

    Returns
    -------
    freqs : list
        List of categories and frequencies.
    """
    n_tokens = len(tokens)
    # Get word counts for each category.
    counts = Counter( cat for tok in tokens for cat in parser(tok) )
    # Convert word counts to frequencies.
    # freqs = { cat: n/n_tokens for cat, n in counts.items() }
    freqs = [ round(100*((counts[c] if c in counts else 0)/n_tokens), n_decimals) for c in categories ]
    return freqs


def liwc_run(
        df,
        dictionary_path,
        export_path,
    ):

    # Open a file to write results to line-by-line.
    with open(export_path, "wt", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile, delimiter="\t", quoting=csv.QUOTE_NONE)

        # Write header.
        column_names = ["text_id"] + category_names
        if SEGMENTED:
            column_names.insert(1, "segment")
        writer.writerow(column_names)

        # loop over txt files
        for fn in tqdm.tqdm(import_fnames, desc="LIWCing all txt files"):
            with open(fn, "rt", encoding="utf-8") as infile:
                txt = infile.read()
            text_id = fname2id(fn)
            if SEGMENTED:
                # txt_segments = segment_txt(txt)
                # if txt_segments is not None:
                if (tokens := tokenize4liwc(txt)) is not None:
                    for i, segment_toks in enumerate(segment(tokens)):
                        results = liwc_tokens(segment_toks)
                        rowdata = [text_id, i+1] + results
                        writer.writerow(rowdata)
            else:
                if (tokens := tokenize4liwc(txt)) is not None:
                    results = liwc_tokens(tokens)
                    rowdata = [text_id] + results
                    writer.writerow(rowdata)




########################################################
# Visualizations
########################################################

def plot_timecourse(
        df,
        temporal_chunk="month",
        timestamp_column="timestamp",
        author_column="author",
        categorical_column=None,
        categorical_order=None,
        palette=None,
        cumulative_twin=True,
    ):
    """Visualize the amount of data over time.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with a single text entry in each row (i.e., long format).
    temporal_chunk : str
         (must be
        one of 'year', 'month', 'week').
    timestamp_column : str
        Name of column containing the text timestamp.
        Must be in ISO-8601 format (e.g., "1995-02-04 22:45:00").
    author_column : str
        Name of column containing the unique author ID.
    palette : dict
        Colors to use if categorical column is used.
    cumulative_twin : bool
        If True, draw a cumulative step histogram on the right axis.
        Useful to visualize total counts.

    Returns
    -------
    fig : matplotlib Figure
        Returns the Figure object.
    hist_df : pd.DataFrame
        A dataframe containing histogram values.

    Notes
    -----
    Visualize the amount of data over time, with the option to
    color by a categorical variable. The visualization should
    take the amount of unique authors into account.
    """

    # Check column arguments.
    assert timestamp_column in df, f"Column {timestamp_column} not found in dataframe."
    assert author_column in df, f"Column {author_column} not found in dataframe."

    # Check content of columns.
    assert temporal_chunk in ["year", "month", "week"], "Temporal chunk must be one of year, month, week."
    assert not (categorical_column is None and palette is not None), "Can't use palette without a categorical variable."

    # Load custom plot settings.
    load_matplotlib_settings()

    # Set defaults.
    figsize = (6, 2)
    hist_kwargs = {
        "alpha": 1,
        "linewidth": 1,
        "edgecolor": "white",
        "histtype": "barstacked",
        "cumulative": False,
        "stacked": True,
    }

    # Pick datetime bins and tick locators.
    readable2freq = dict(year="y", month="M", week="W")
    freq = readable2freq[temporal_chunk]
    xmin = df[timestamp_column].min()
    xmax = df[timestamp_column].max()
    bins = pd.date_range(start=xmin, end=xmax, freq=freq, normalize=True)
    hist_kwargs.update({"bins": bins})
    if temporal_chunk == "year":
        major_xlocator = plt.matplotlib.dates.YearLocator()
        minor_xlocator = plt.matplotlib.dates.YearLocator()
    elif temporal_chunk == "month":
        major_xlocator = plt.matplotlib.dates.YearLocator()
        minor_xlocator = plt.matplotlib.dates.MonthLocator()
    elif temporal_chunk == "week":
        major_xlocator = plt.matplotlib.dates.MonthLocator()
        minor_xlocator = plt.matplotlib.dates.WeekLocator()
    # major_xformatter = plt.matplotlib.dates.DateFormatter('%d.%m.%y')

    fig, ax = plt.subplots(figsize=figsize)

    if categorical_column is None:
        hist_data = df[timestamp_column]
    else:
        labels = df[categorical_column].unique().tolist()
        if categorical_order is not None:
            assert sorted(labels) == sorted(categorical_order)
        else:
            categorical_order = sorted(labels)
        hist_data = df.groupby(categorical_column
            )[timestamp_column].apply(list
            ).loc[categorical_order].tolist()
        hist_kwargs.update({"label": categorical_order})
        if palette is not None:
            colors = [ palette[k] for k in categorical_order ]
            hist_kwargs.update({"color": colors})

    counts, bins, bars = ax.hist(hist_data, **hist_kwargs)

    # Make a dataframe of histogram values.
    column_names = categorical_order if categorical_column else ["count"]
    hist_df = pd.DataFrame(counts.T, columns=column_names)
    hist_df.insert(0, "bin_start", bins[1:])
    hist_df.insert(1, "bin_stop", bins[:-1])

    ax.xaxis.set(major_locator=major_xlocator,
                 minor_locator=minor_xlocator)
    ylabel = fr"$n$ posts per {temporal_chunk}"
    ax.set(ylabel=ylabel)

    # Legend.
    ax.legend(loc="upper left")

    ##### Twin axis
    if cumulative_twin:
        hist_kwargs.update({
            "cumulative": True,
            "histtype": "step",
        })
        axt = ax.twinx()
        counts, bins, bars = axt.hist(hist_data, **hist_kwargs)

    return fig, hist_df