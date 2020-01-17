from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

text_str = '''
As an interdisciplinary eld, machine learning shares common threads with the
mathematical elds of statistics, information theory, game theory, and optimization.
It is naturally a subeld of computer science, as our goal is to program
machines so that they will learn. In a sense, machine learning can be viewed as
a branch of AI (Articial Intelligence), since, after all, the ability to turn experience
into expertise or to detect meaningful patterns in complex sensory data
is a cornerstone of human (and animal) intelligence. However, one should note
that, in contrast with traditional AI, machine learning is not trying to build
automated imitation of intelligent behavior, but rather to use the strengths and special abilities of computers to complement human intelligence, often performing
tasks that fall way beyond human capabilities. For example, the ability to
scan and process huge databases allows machine learning programs to detect
patterns that are outside the scope of human perception.
The component of experience, or training, in machine learning often refers
to data that is randomly generated. The task of the learner is to process such
randomly generated examples toward drawing conclusions that hold for the environment
from which these examples are picked. This description of machine
learning highlights its close relationship with statistics. Indeed there is a lot in
common between the two disciplines, in terms of both the goals and techniques
used. There are, however, a few signicant dierences of emphasis; if a doctor
comes up with the hypothesis that there is a correlation between smoking and
heart disease, it is the statistician's role to view samples of patients and check
the validity of that hypothesis (this is the common statistical task of hypothesis
testing). In contrast, machine learning aims to use the data gathered from
samples of patients to come up with a description of the causes of heart disease.
The hope is that automated techniques may be able to gure out meaningful
patterns (or hypotheses) that may have been missed by the human observer.
In contrast with traditional statistics, in machine learning in general, and
in this book in particular, algorithmic considerations play a major role. Machine
learning is about the execution of learning by computers; hence algorithmic
issues are pivotal. We develop algorithms to perform the learning tasks and
are concerned with their computational eciency. Another dierence is that
while statistics is often interested in asymptotic behavior (like the convergence
of sample-based statistical estimates as the sample sizes grow to innity), the
theory of machine learning focuses on nite sample bounds. Namely, given the
size of available samples, machine learning theory aims to gure out the degree
of accuracy that a learner can expect on the basis of such samples.
There are further dierences between these two disciplines, of which we shall
mention only one more here. While in statistics it is common to work under the
assumption of certain presubscribed data models (such as assuming the normality
of data-generating distributions, or the linearity of functional dependencies),
in machine learning the emphasis is on working under a \distribution-free" setting,
where the learner assumes as little as possible about the nature of the
data distribution and allows the learning algorithm to gure out which models
best approximate the data-generating process. A precise discussion of this issue
requires some technical preliminaries, and we will come back to it later in the
book, and in particular in Chapter 5.
'''

def create_frequency_table(text_string) -> dict:
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()
    
    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word]+= 1
        else:
            freqTable[word] = 1
            
    return freqTable

def score_sentences(sentences, freqTable) -> dict:
    sentenceValue = dict()
    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]
        if sentence[:10] in sentenceValue:
            sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] / word_count_in_sentence_except_stop_words
    return sentenceValue

def find_average_score(sentenceValue) -> int:
    sumValue = 0
    for entry in sentenceValue:
        sumValue += sentenceValue[entry]
        
    average = (sumValue/ len(sentenceValue))
    
    return average

def generate_summary(sentences,sentenceValue, threshold):
    sentence_count = 0
    summary = ''
    
    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]]>=(threshold):
            summary += ""+ sentence
            sentence_count += 1  
    return summary


def run_word_frequency_summarization(text):
    
    freq_table = create_frequency_table(text)
    
    sentences = sent_tokenize(text)
    
    sentence_score = score_sentences(sentences, freq_table)
    
    threshold = find_average_score(sentence_score)
    
    summary = generate_summary(sentences, sentence_score, 1.3 * threshold)
    
    
    return summary


        
        
        





         
    
    
    