import math
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords

text_str = '''
Those Who Are Resilient Stay In The Game Longer
â€œOn the mountains of truth you can never climb in vain: either you will reach a point higher up today, or you will be training your powers so that you will be able to climb higher tomorrow.â€�â€Šâ€”â€ŠFriedrich Nietzsche
Challenges and setbacks are not meant to defeat you, but promote you. However, I realise after many years of defeats, it can crush your spirit and it is easier to give up than risk further setbacks and disappointments. Have you experienced this before? To be honest, I donâ€™t have the answers. I canâ€™t tell you what the right course of action is; only you will know. However, itâ€™s important not to be discouraged by failure when pursuing a goal or a dream, since failure itself means different things to different people. To a person with a Fixed Mindset failure is a blow to their self-esteem, yet to a person with a Growth Mindset, itâ€™s an opportunity to improve and find new ways to overcome their obstacles. Same failure, yet different responses. Who is right and who is wrong? Neither. Each person has a different mindset that decides their outcome. Those who are resilient stay in the game longer and draw on their inner means to succeed.

Iâ€™ve coached mummy and mom clients who gave up after many years toiling away at their respective goal or dream. It was at that point their biggest breakthrough came. Perhaps all those years of perseverance finally paid off. It was the 19th Centuryâ€™s minister Henry Ward Beecher who once said: â€œOneâ€™s best success comes after their greatest disappointments.â€� No one knows what the future holds, so your only guide is whether you can endure repeated defeats and disappointments and still pursue your dream. Consider the advice from the American academic and psychologist Angela Duckworth who writes in Grit: The Power of Passion and Perseverance: â€œMany of us, it seems, quit what we start far too early and far too often. Even more than the effort a gritty person puts in on a single day, what matters is that they wake up the next day, and the next, ready to get on that treadmill and keep going.â€�

I know one thing for certain: donâ€™t settle for less than what youâ€™re capable of, but strive for something bigger. Some of you reading this might identify with this message because it resonates with you on a deeper level. For others, at the end of their tether the message might be nothing more than a trivial pep talk. What I wish to convey irrespective of where you are in your journey is: NEVER settle for less. If you settle for less, you will receive less than you deserve and convince yourself you are justified to receive it.


â€œTwo people on a precipice over Yosemite Valleyâ€� by Nathan Shipps on Unsplash
Develop A Powerful Vision Of What You Want
â€œYour problem is to bridge the gap which exists between where you are now and the goal you intend to reach.â€�â€Šâ€”â€ŠEarl Nightingale
I recall a passage my father often used growing up in 1990s: â€œDonâ€™t tell me your problems unless youâ€™ve spent weeks trying to solve them yourself.â€� That advice has echoed in my mind for decades and became my motivator. Donâ€™t leave it to other people or outside circumstances to motivate you because you will be let down every time. It must come from within you. Gnaw away at your problems until you solve them or find a solution. Problems are not stop signs, they are advising you that more work is required to overcome them. Most times, problems help you gain a skill or develop the resources to succeed later. So embrace your challenges and develop the grit to push past them instead of retreat in resignation. Where are you settling in your life right now? Could you be you playing for bigger stakes than you are? Are you willing to play bigger even if it means repeated failures and setbacks? You should ask yourself these questions to decide whether youâ€™re willing to put yourself on the line or settle for less. And thatâ€™s fine if youâ€™re content to receive less, as long as youâ€™re not regretful later.

If you have not achieved the success you deserve and are considering giving up, will you regret it in a few years or decades from now? Only you can answer that, but you should carve out time to discover your motivation for pursuing your goals. Itâ€™s a fact, if you donâ€™t know what you want youâ€™ll get what life hands you and it may not be in your best interest, affirms author Larry Weidel: â€œWinners know that if you donâ€™t figure out what you want, youâ€™ll get whatever life hands you.â€� The key is to develop a powerful vision of what you want and hold that image in your mind. Nurture it daily and give it life by taking purposeful action towards it.

Vision + desire + dedication + patience + daily action leads to astonishing success. Are you willing to commit to this way of life or jump ship at the first sign of failure? Iâ€™m amused when I read questions written by millennials on Quora who ask how they can become rich and famous or the next Elon Musk. Success is a fickle and long game with highs and lows. Similarly, there are no assurances even if youâ€™re an overnight sensation, to sustain it for long, particularly if you donâ€™t have the mental and emotional means to endure it. This means you must rely on the one true constant in your favour: your personal development. The more you grow, the more you gain in terms of financial resources, status, successâ€Šâ€”â€Šsimple. If you leave it to outside conditions to dictate your circumstances, you are rolling the dice on your future.

So become intentional on what you want out of life. Commit to it. Nurture your dreams. Focus on your development and if you want to give up, know whatâ€™s involved before you take the plunge. Because I assure you, someone out there right now is working harder than you, reading more books, sleeping less and sacrificing all they have to realise their dreams and it may contest with yours. Donâ€™t leave your dreams to chance.
'''

def create_frequency_table(text_string) -> dict:
    stopWords_list = set(stopwords.words("english"))
    words_list = word_tokenize(text_string)
    
    ps = PorterStemmer()
    freqTable = dict()
    for word in words_list:
        word = ps.stem(word)
        if word in stopWords_list:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    return freqTable

def create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords_list = set(stopwords.words("english"))
    ps = PorterStemmer()
    
    for sent in sentences:
        freqTable = {}
        words_list = word_tokenize(sent)
        for word in words_list:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords_list:
                continue
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1 
        
        frequency_matrix[sent[:15]] = freqTable
        
    return frequency_matrix


def create_tf_matrix(freq_matrix):
    tf_matrix = {}
    for sent, f_table in freq_matrix.items():
        tf_table = {}
        count_words_in_sentence = len(f_table)
        for word , count in f_table.items():
            tf_table[word] = count /count_words_in_sentence
        tf_matrix[sent] = tf_table
    return tf_matrix

def create_documents_per_words(freq_matrix):
    word_per_doc_table = {}
    
    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1
    return word_per_doc_table

def create_idf_matrix(freq_matrix,count_doc_per_words,total_documents):
    idf_matrix = {}
    
    for sent,f_table in freq_matrix.items():
        idf_table = {}
        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))
        idf_matrix[sent] = idf_table
    return idf_matrix

def create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}
    
    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word1, value1), (word2, value2) in zip(f_table1.items(),f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)
        
        tf_idf_matrix[sent1] = tf_idf_table
        
    return tf_idf_matrix
            

def score_sentences(tf_idf_matrix) -> dict:
    sentenceValue = {}       
    
    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0
        count_words_in_sentence = len(f_table)
        
        for word, score in f_table.items():
            total_score_per_sentence += score
        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence
    return sentenceValue   
    
def find_average_score(sentenceValue) -> int:
    sumValues = 0
    
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]
    
    average = (sumValues / len(sentenceValue))
    
    return average

def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''
    
    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1
    return summary
    

def run_TF_IDF_summarization(text):
    # 1 Sentence Tokenize
    sentences = sent_tokenize(text)
    total_documents = len(sentences)
    
    # 2 Create the Frequency matrix of the words in each sentence.
    freq_matrix = create_frequency_matrix(sentences)
    
    # 3 Calculate TermFrequency and generate a matrix
    tf_matrix = create_tf_matrix(freq_matrix)
    
    # 4 creating table for documents per words
    count_doc_per_words = create_documents_per_words(freq_matrix)
    
    # 5 Calculate IDF and generate a matrix
    idf_matrix = create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    
    # 6 Calculate TF-IDF and generate a matrix
    tf_idf_matrix = create_tf_idf_matrix(tf_matrix, idf_matrix)
    
    # 7 Important Algorithm: score the sentences
    sentence_scores = score_sentences(tf_idf_matrix)
    
    # 8 Find the threshold
    threshold = find_average_score(sentence_scores)
    
    # 9 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 1.3 * threshold)
    return summary
    

    

    
    
    
    
    
            