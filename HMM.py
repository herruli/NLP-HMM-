import nltk
import sys
from nltk.corpus import brown
from collections import Counter
brownTagsWords = []
brownTagsTrained = []
overallAccuracy= []

def pannelOperation():
    print('1 for normal HMM (No TagSize reduce) operation')
    print('2 for reduce All TagSize HMM operation')
    print('Enter operation')
    global operation
    operation = input()
    tagSentenceOperation(operation)
    return operation
def tagSentenceOperation(operation):
    if operation ==1:
        tagsentence()
    if operation == 2:
        reduceALLTagSize()

def tagsentence():
    for sent in brown.tagged_sents()[0:50000]:
        # add /s at the beginning
        brownTagsWords.append(("/s", "/s"))
        # change the sequence to tag,word for easier to use in HMM
        brownTagsWords.extend([(tag,word) for (word,tag) in sent])
        # then /e
        brownTagsWords.append(("/e", "/e"))

def reduceALLTagSize():
    print('2 for reduce All TagSize HMM operation')
    for sent in brown.tagged_sents()[0:50000]:
        # add /s at the beginning
        brownTagsWords.append(("/s", "/s"))
        # change the sequence to tag,word for easier to use in HMM
        brownTagsWords.extend([(tag[:2],word) for (word,tag) in sent])
        # then /e
        brownTagsWords.append(("/e", "/e"))


def unknownWordsTraining():
    #counts will find the words frequency and store as dictionary
    counts= Counter([word for (tag,word) in brownTagsWords])
    #the following for loop will loop and change the word of frequency with 1 to 'UNK'
    for idx,word in enumerate(brownTagsWords):
        if counts.get(word[1]) == 1:
            temp=list(brownTagsWords[idx])
            temp[1]='UNK'
            brownTagsWords[idx]=tuple(temp)

def unknownWordsTesting(sentence):
    #counts will find the words frequency with more than 2
    counts= Counter([word for (tag,word) in brownTagsWords])
    countsWord= list(counts.keys())
    #it will show the words can be found in sentence but not in the training set
    unknownWords=(set(sentence).difference(set(countsWord)))
    #it will change the words in the sentence which cannot be found in training set to 'UNK'
    for wordsunknown in unknownWords:
        for idx, word in enumerate(sentence):
            if (wordsunknown == word):
                sentence[idx] = 'UNK'

def hmm():
    # calculate the frequency distribution for the tags, words
    conditionFreqDisttag = nltk.ConditionalFreqDist(brownTagsWords)
    # calculate the probability distribution for the tags,words by using Maximum Likelihood Estimation
    global conditionProbdistTag
    conditionProbdistTag = nltk.ConditionalProbDist(conditionFreqDisttag, nltk.MLEProbDist)
    #extract the tags and train it in HMM
    global brownTagsTrained
    brownTagsTrained = [tag for (tag, word) in brownTagsWords ]
    # calculate the bigram frequency distribution for the tags
    #calculate the P(ti|ti-1)=C(ti-1ti)+1/C(ti-1)+V as V stands for the tags there are in the corpus
    freqDistTags= nltk.ConditionalFreqDist(nltk.bigrams(brownTagsTrained))
    # calculate the probability distribution for the tags by using Maximum Likelihood Estimation
    # the HMM is now trained
    global probdistTags
    probdistTags= nltk.ConditionalProbDist(freqDistTags, nltk.LaplaceProbDist, bins=len(brownTagsTrained))
    viterbi()


def viterbi():
    #extract all tags appeared in the first 50000 sentences
    allDistinctTags = set(brownTagsTrained)
    #loop the sentence through the viterbi algorithm
    for sent in brown.tagged_sents()[50001:50500]:
        #extract the words from the sentence and store in a list
        sentence = [word for (word, tag) in sent]
        #extract the tags from the sentence and store in a list for later comparison
        originaltag = [tag for (word, tag) in sent]
        reduceOriginaltag= [tag[:2] for (word, tag) in sent]
        #run the sentence and compare with the words in training set see if appear or not. If cannot find, change it to UNK
        unknownWordsTesting(sentence)
        viterbiList = []
        backpointer = []
        bestTagSequence = []
        firstViterbi(allDistinctTags,viterbiList,backpointer,sentence)
        otherViterbi(allDistinctTags,viterbiList,backpointer,sentence)
        #when the sentence is finished, it will find the probability of the tag with the next tag of "/e"
        prevViterbiList = viterbiList[-1]
        #viterbi[qF,n+1]=max(viterbi[q',n]*alpha(q',qF))
        bestPreviousTag = max(prevViterbiList.keys(),key = lambda prevoustag: prevViterbiList[prevoustag] * probdistTags[prevoustag].prob("/e"))
        bestTagSequence = [ "/e", bestPreviousTag ]
        bestTag = bestPreviousTag
        backpointer.reverse()
        #the probability of the most likely sequence of states are in the bestTagSequence
        for pointer in backpointer:
            bestTagSequence.append(pointer[bestTag])
            bestTag = pointer[bestTag]
        bestTagSequence.reverse()
        bestTagSequence.pop(0)
        bestTagSequence.pop(-1)
        resultTag(sentence,bestTagSequence,originaltag,reduceOriginaltag)



def firstViterbi(allDistinctTags,viterbiList,backpointer,sentence):
#The following is to calculate the initial state of viterbi
    firstViterbiList = {}
    firstBackPointer = {}
    for tag in allDistinctTags:
        if tag == "/s": continue
        #viterbi[q,1]=alpha(q0,q)*beta(q,w1)
        firstViterbiList[tag] = probdistTags["/s"].prob(tag) * conditionProbdistTag[tag].prob(sentence[0])
        firstBackPointer[tag] = "/s"
    #this will save the tag which has the best result
    viterbiList.append(firstViterbiList)
    #backpointer will be used to record how probabiliy in the table arises
    backpointer.append(firstBackPointer)
    return viterbiList,backpointer

def otherViterbi(allDistinctTags,viterbiList,backpointer,sentence):
    #The following is to calculate the remaining states except the start and the end
    for wordindex in range(1, len(sentence)):
        thisViterbiList = {}
        thisBackPointer = {}
    #prevViterbiList will take the last item to become viterbi[q,i-1] for further use
        prevViterbiList = viterbiList[-1]
        for tag in allDistinctTags:
            if tag == "/s": continue
    # this will find the best previous tag from the tag list
            #bestPreviousTag will find the maximum probability tag by calculating the viterbi[q,i]=maxq'viterbi[q',i-1]*alpha(q',q)*beta(q,wi)
            bestPreviousTag = max(prevViterbiList.keys(),key = lambda prevoustag: prevViterbiList[prevoustag] * probdistTags[prevoustag].prob(tag) * conditionProbdistTag[tag].prob(sentence[wordindex]))
            thisViterbiList[tag] = prevViterbiList[bestPreviousTag] * probdistTags[bestPreviousTag].prob(tag) * conditionProbdistTag[tag].prob(sentence[wordindex])
            thisBackPointer[tag] = bestPreviousTag
        viterbiList.append(thisViterbiList)
        backpointer.append(thisBackPointer)
    return viterbiList,backpointer

def resultTag(sentence,bestTagSequence,originaltag,reduceOriginaltag):
    accuracy= 0
    count = 1
    print('The first sentence')
    print(count)
    count +=1
    print('The best tag sequence')
    print(list(bestTagSequence))
    print("\n")
    if operation == 2:
        originaltag= reduceOriginaltag
        print('The tag in brown corpus')
        print(list(originaltag))
    else:
        print('The tag in brown corpus')
        print(list(originaltag))
    for i in range(0, len(originaltag)):
        if originaltag[i] == bestTagSequence[i]:
            accuracy +=1
    print ("The accuracy is", float(accuracy)/float(len(originaltag))*100)
    overallAccuracy.append(float(accuracy)/float(len(originaltag))*100)

def overAll(overallAccuracy):
    print("Overall Accuracy equals", sum(overallAccuracy)/len(overallAccuracy))

pannelOperation()
unknownWordsTraining()
hmm()
overAll(overallAccuracy)
