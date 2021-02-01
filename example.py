from redditnlp import RedditWordCounter, TfidfCorpus
import urllib
import os
from collections import deque

######################
# SETTINGS
######################

USERNAME = '6WrnXsVcLIA2Fw' #'hapinator1954'  # Change this to your username
CLIENT_SECRET = 'fGUKP3WyXgIyymflYLHQvBrehLPt0g'
SAVE_DIR = 'tfidf_corpus'
CORPUS_FILE = 'corpus.json'
COMMENTS_PER_SUBREDDIT = 1000
SUBREDDITS = [
    'wallstreetbets','thetagang','robinhoodpennystocks','YOLO','gaybears'
]
# 'thetagang','robinhoodpennystocks','YOLO'
###########################
# VOCABULARY ANALYTICS
###########################

# Load tickers
application_root = os.path.dirname(__file__)
all_tickers = os.path.join(application_root, 'redditnlp/words/all_tickers.csv')
with open(all_tickers, 'r') as tickers_file:
    all_tickers = set([word.strip('\n') for word in tickers_file.readlines()])

def ticker_filter(item_to_filter):
    if item_to_filter in all_tickers:
        return True
    else:
        return False

def get_subreddit_vocabularies():
    # Initialise Reddit word counter instance
    reddit_counter = RedditWordCounter(USERNAME, CLIENT_SECRET)

    # Initialise tf-idf corpus instance
    corpus_path = os.path.join(SAVE_DIR, CORPUS_FILE)
    comment_corpus = TfidfCorpus(corpus_path)

    # Extract the vocabulary for each of the subreddits specified
    subreddit_queue = deque([subreddit for subreddit in SUBREDDITS])
    while len(subreddit_queue) > 0:
        subreddit = subreddit_queue.popleft()

        try:
            vocabulary = reddit_counter.subreddit_comments_new(subreddit, limit=COMMENTS_PER_SUBREDDIT)
        except urllib.error.HTTPError as err:
            print (err)
            # Add subreddit back into queue
            subreddit_queue.append(subreddit)
            continue

        comment_corpus.add_document(vocabulary, subreddit)
        comment_corpus.save()

    return comment_corpus, corpus_path


def save_subreddit_top_terms(corpus):
    # Save the top terms for each subreddit in a text file
    save_path = os.path.join(SAVE_DIR, 'top_words.txt')
    for document in corpus.get_document_list():
        top_terms = corpus.get_top_terms(document, num_terms=500)
        top_terms = sorted(top_terms.items(), key=lambda x: x[1], reverse=True)
        with open(save_path, 'a', encoding="utf-8") as f:
            f.write(
                document +  #.encode('utf-8') +
                '\n' +
                 '\n'.join(['{0}, {1}'.format(term.encode('utf-8'), weight) for term, weight in top_terms]) +
                 '\n\n')

    return save_path

def save_subreddit_tickers(corpus):
    # Save the top terms for each subreddit in a text file
    tickers_path = os.path.join(SAVE_DIR, 'ticker_list.txt')
    for document in corpus.get_document_list():
        tickers = corpus.get_top_terms(document, num_terms=500)
        tickers = sorted(tickers.items(), key=lambda x: x[1], reverse=True)

        with open(tickers_path, 'a', encoding="utf-8") as g:
            g.write(
                ','.join(
                    filter(
                        ticker_filter,
                            ['{0}'.format((str(t[0]).upper())) for t in tickers]

                    )
                )
            )
    return tickers_path

def get_swearword_counts(corpus):
    with open('redditnlp/words/swearwords_english.txt', 'r') as f:
        swearwords = [word.strip('\n') for word in f.readlines()]

    swearword_counts = dict()
    for document in corpus.get_document_list():
        swearword_counts[document] = corpus.count_words_from_list(document, swearwords)
    return swearword_counts


def get_vocabulary_sophistication(corpus):
    mean_word_lengths = dict()
    for document in corpus.get_document_list():
        mean_word_lengths[document] = corpus.get_mean_word_length(document)
    return mean_word_lengths

# Extract their word counts
corpus, corpus_path = get_subreddit_vocabularies()
print ('TF-IDF corpus saved to %s' % corpus_path)

# Get the top words by subreddit
top_terms_path = save_subreddit_top_terms(corpus)
print ('Top terms saved to %s' % corpus_path)

# Get the top words by subreddit
ticker_path = save_subreddit_tickers(corpus)
print ('Top terms saved to %s' % ticker_path)

# Get the swearword frequency
swearword_frequency = get_swearword_counts(corpus)
print ('Normalized swearword frequency:')
for subreddit, frequency in swearword_frequency.items():
    print ('%s, %s' % (subreddit, frequency))

# Get the average word length
print ('\nAverage word length by subreddit:')
word_lengths = get_vocabulary_sophistication(corpus)
for subreddit, frequency in word_lengths.items():
    print ('%s, %s' % (subreddit, frequency))

#######################
# MACHINE LEARNING DEMO
#######################

# Collect the comments for a particular user and determine which subreddit their comments best match up with
counter = RedditWordCounter(USERNAME,CLIENT_SECRET)
corpus = TfidfCorpus(os.path.join(SAVE_DIR, CORPUS_FILE))

user_comments = counter.user_comments('DeepFuckingValue')
corpus.train_classifier(classifier_type='LinearSVC', tfidf=True)
print (corpus.classify_document(user_comments))