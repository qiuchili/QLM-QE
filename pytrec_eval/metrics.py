import math

__author__ = 'alberto'


# Contains all metrics.
# A metric is a function f(TrecRun, QRels, detailed=False) that returns a double.
# If detailed is True only the aggregated score among all topics is returned (a double);
# otherwise, a pair (aggregatedScore, details) where
# details is a dictionary details[topicID] = score is returned.


def precision(run, qrels, detailed=False):
    """Computes average precision among all entities."""
    details = {}
    avg = 0
    for topicId in qrels.allJudgements:
        if topicId in run.entries:
            entryList = run.entries[topicId]
            numRelevant = len([docId for docId, score, _ in entryList
                               if qrels.isRelevant(topicId, docId)])
            numReturned = len(entryList)
            details[topicId] = numRelevant / numReturned
            avg += numRelevant / numReturned
        else:
            details[topicId] = 0
            avg += 0
    numTopics = qrels.getNTopics()
    return avg / numTopics if not detailed else (avg / numTopics, details)

def precisionAt(rank):
    """
    Computes precision@rank. Returns a function that,
    satisfies the requirements defined on the top of this module and that
    computes precision@rank.
    """

    def precisionAtRank(run, qrels, detailed=False):
        details = {}
        avg = 0
        for topicId, entryList in run.entries.items():
            numRelevant = len([docId for docId, score, _ in entryList[0: rank]
                               if qrels.isRelevant(topicId, docId)])
            details[topicId] = numRelevant / rank
            avg += numRelevant / rank
        numtopics = qrels.getNTopics()
        return avg / numtopics if not detailed else (avg / numtopics, details)

    return precisionAtRank

def recall(run, qrels, detailed=False):
    """Computes recall"""
    details = {}
    avg = 0
    nTopicsWRelevant = 0
    for topicId in qrels.allJudgements:
        numRelevant = qrels.getNRelevant(topicId)
        if topicId in run.entries:
            entryList = run.entries[topicId]
            numRelevantFound = len([docId for docId, score, _ in entryList
                                    if qrels.isRelevant(topicId, docId)])
            if numRelevant > 0:
                details[topicId] = numRelevantFound / numRelevant
                avg += numRelevantFound / numRelevant
                nTopicsWRelevant += 1
                # ignore queries without relevant docs is 1
        else:
            details[topicId] = 0
            avg += 0
            if numRelevant > 0: nTopicsWRelevant += 1
    # numtopics = qrels.getNTopics()
    return avg / nTopicsWRelevant if not detailed else (avg / nTopicsWRelevant, details)

def recallAt(rank):
    """
    Computes precision@rank. Returns a function that,
    satisfies the requirements defined on the top of this module and that
    computes precision@rank.
    """

    def recallAtRank(run, qrels, detailed=False):
        details = {}
        avg = 0
        nTopicsWRelevant = 0
        for topicId in qrels.allJudgements:
            numRelevant = qrels.getNRelevant(topicId)
            if topicId in run.entries:
                entryList = run.entries[topicId]
                numRelevantFound = len([docId for docId, score, _ in entryList[0:rank] if qrels.isRelevant(topicId, docId)])
                if numRelevant > 0:
                    details[topicId] = numRelevantFound / numRelevant
                    avg += numRelevantFound / numRelevant
                    nTopicsWRelevant += 1
                    # ignore queries without relevant docs is 1
            else:
                details[topicId] = 0
                avg += 0
                if numRelevant > 0: nTopicsWRelevant += 1
        # numtopics = qrels.getNTopics()
        return avg / nTopicsWRelevant if not detailed else (avg / nTopicsWRelevant, details)
    return recallAtRank



def meanAvgPrec(run, qrels, detailed=False):
    """Computes average precision."""
    details = {}
    avg = 0
    for topicId, entryList in run.entries.items():
        sumPrec = numRel = 0
        for (rank, (docId, score, _)) in enumerate(entryList, start=1):
            if qrels.isRelevant(topicId, docId):
                numRel += 1
                sumPrec += numRel / rank
        totRelevant = qrels.getNRelevant(topicId)
        # if totRelevant == 0: print(topicId)
        ap = sumPrec / totRelevant if totRelevant > 0 else 0
        avg += ap
        details[topicId] = ap
    numtopics = qrels.getNTopics()
    return avg / numtopics if not detailed else (avg / numtopics, details)


def meanAvgPrecAt(rank):
    """
    Computes precision@rank. Returns a function that,
    satisfies the requirements defined on the top of this module and that
    computes precision@rank.
    """

    def meanAvgPrecAtRank(run, qrels, detailed=False):
        details = {}
        avg = 0
        for topicId, entryList in run.entries.items():
            sumPrec = numRel = 0
            for (oneRank, (docId, score, _)) in enumerate(entryList[0:rank], start=1):
                if qrels.isRelevant(topicId, docId):
                    numRel += 1
                    sumPrec += numRel / oneRank
            totRelevant = qrels.getNRelevant(topicId)
            # if totRelevant == 0: print(topicId)
            ap = sumPrec / numRel if numRel > 0 else 0
            avg += ap
            details[topicId] = ap
        numtopics = qrels.getNTopics()
        return avg / numtopics if not detailed else (avg / numtopics, details)
    return meanAvgPrecAtRank



def ndcg(run, qrels, detailed=False):
    """
    Computes NDCG using the formula
    DCG_p = rel_1 + \sum_{i = 2}^p( rel_i / log_2(i) )
    Where p is the number of entries of the given run for
    a certain topic, and rel_i is the relevance score of the
    document at rank i.
    """
    details = {}
    avg = 0
    for topicId, entryList in run.entries.items():
        relevancesByRank = qrels.getRelevanceScores(topicId, [doc for (doc, _, _) in entryList])
        sumdcg = relevancesByRank[0] + sum([relScore / math.log2(rank)
                                            for rank, relScore in enumerate(relevancesByRank[1:], start=2)])
        relevancesByRank.sort(reverse=True)  # sort the relevance list descending order
        sumIdcg = relevancesByRank[0] + sum([relScore / math.log2(rank)
                                             for rank, relScore in enumerate(relevancesByRank[1:], start=2)])
        if sumIdcg == 0:
            details[topicId] = 0
        else:
            details[topicId] = sumdcg / sumIdcg
            avg += sumdcg / sumIdcg
    numtopics = qrels.getNTopics()
    return avg / numtopics if not detailed else (avg / numtopics, details)

def ndcgAt(rank):
    """
    Computes precision@rank. Returns a function that,
    satisfies the requirements defined on the top of this module and that
    computes precision@rank.
    """


    def ndcgAtRank(run, qrels, detailed=False):
        details = {}
        avg = 0
        for topicId, entryList in run.entries.items():
            relevancesByRank = qrels.getRelevanceScores(topicId, [doc for (doc, _, _) in entryList[1:rank]])
            sumdcg = relevancesByRank[0] + sum([relScore / math.log2(onerank) for onerank, relScore in enumerate(relevancesByRank, start=2)])

            relevancesByRank.sort(reverse=True)  # sort the relevance list descending order
            sumIdcg = relevancesByRank[0] + sum([relScore / math.log2(onerank)
                                             for onerank, relScore in enumerate(relevancesByRank, start=2)])
            if sumIdcg == 0:
                details[topicId] = 0
            else:
                details[topicId] = sumdcg / sumIdcg
                avg += sumdcg / sumIdcg
        numtopics = qrels.getNTopics()
        return avg / numtopics if not detailed else (avg / numtopics, details)
    return ndcgAtRank


STD_METRICS = [meanAvgPrec, ndcg]
