from pytrec_eval import *
import numpy as np
import os

eval_dir = 'sessiontrack/2014'
qrel_path = os.path.join(eval_dir,'qrel_per_session.txt')
run_path_original = os.path.join(eval_dir,'indri_ranking.txt')
run_path_reranking = os.path.join(eval_dir, 'reranking.txt')
# run_path_initial = os.path.join(eval_dir,'initial_ranking.txt')
run_origin = TrecRun(run_path_original)
qrels = QRels(qrel_path)
# run_initial = TrecRun(run_path_initial)
run_reranking = TrecRun(run_path_reranking)

# a = np.zeros((3,3))
# a[:2,:2] = np.random.random((2,2))
# print(a)
print(evaluate(run_origin, qrels, measures=[meanAvgPrecAt(10), ndcgAt(1),ndcgAt(5), ndcgAt(10)]))

# print(evaluate(run_initial, qrels, measures=[precisionAt(10),ndcgAt(1), ndcgAt(5), ndcgAt(10)]))

print(evaluate(run_reranking, qrels, measures=[meanAvgPrecAt(10),ndcgAt(1), ndcgAt(5), ndcgAt(10)]))
# if not os.path.exists('asa'):
#     print("asa")
# a = [1,2,3]
# b = np.diag(a)
# print(b/np.trace(b))
