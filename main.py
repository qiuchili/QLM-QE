# -*- coding:utf-8 -*-
from indri import IndriAPI
import os
import codecs
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from itertools import combinations
import string
import numpy as np
from GQLM import GQLM
from QLM_utils import vn_divergence
from pytrec_eval import *
from collections import OrderedDict
from params import Params

def write_result_to_file(doc_score_dict, session_id, file):
	sorted_doc_list = sorted(doc_score_dict, key=doc_score_dict.__getitem__, reverse = True)

	rank = 1
	for document_id in sorted_doc_list:
		score = np.real(doc_score_dict[document_id])
		line = session_id+'\t'+'Q0'+'\t'+document_id+ '\t'+ str(rank)+ '\t' + str(score) + '\t'+ 'Exp'+'\n'
		rank = rank+1
		file.write(line)
	return(sorted_doc_list)

def get_quantum_event_count(indri, dependency_dict, doc_array_list, window_size):
	total_dependency_count = 0
	for doc_array in doc_array_list:
		for one_dependency in dependency_dict:
			total_dependency_count = total_dependency_count+indri.get_co_occur_count(one_dependency, doc_array, window_size)

	return(total_dependency_count)

def extract_quantum_events(indri, dependency_dict, doc_array_list, window_size, normalized = True):
	proj_dict = {}
	total_dependency_count = 0
	for doc_array in doc_array_list:
		for one_dependency in dependency_dict:
			total_dependency_count = total_dependency_count+indri.get_co_occur_count(one_dependency, doc_array, window_size)

	if total_dependency_count == 0:
		return(proj_dict)
	for doc_array in doc_array_list:
		for one_dependency in dependency_dict:
			dependency_count = indri.get_co_occur_count(one_dependency, doc_array, window_size)
			if normalized:
				weight = dependency_count/total_dependency_count
			else:
				weight = dependency_count
			if dependency_count > 0:
				if one_dependency in proj_dict:
					proj_dict[one_dependency][0] = proj_dict[one_dependency][0]+weight
				else:
					projector = dependency_dict[one_dependency]
					proj_dict[one_dependency] = [weight, projector]
	return(proj_dict)

def generate_dependency_list(term_list, query_array,max_dependency_count):
	max_val = min(len(term_list), max_dependency_count)
	dependency_dict = {}
	for count in range(max_val):
		for comb in combinations(range(len(term_list)), count+1):
			dependency = ''
			vector = np.zeros(len(term_list))
			for one_index in comb:
				dependency = dependency + term_list[one_index]+' '
				vector[one_index] = query_array[one_index]
			dependency = dependency.strip()
			projector = np.outer(vector, vector)/np.inner(vector, vector)
			dependency_dict[dependency] = projector

	return(dependency_dict)

def file_evaluate(qrel_path, run_path, eval_path):
	run = TrecRun(run_path)
	qrels = QRels(qrel_path)
	run = TrecRun(run_path)
	qrels = QRels(qrel_path)
	performances = evaluate(run, qrels, measures=[ndcgAt(5), ndcgAt(10),meanAvgPrec])
	with codecs.open(eval_path,'w') as eval_file:
		eval_file.write(performances)

def extract_session_history(indri,interaction_list):
	interaction_query_list = []
	interaction_term_list = []
	for one_interaction in interaction_list:
		interaction_text = preprocess(one_interaction)
		interaction_query = indri.get_query_array(interaction_text)
		for term in interaction_query:
			if not term in interaction_term_list:
				interaction_term_list.append(term)
		interaction_query_list.append(interaction_query)
	return(interaction_query_list, interaction_term_list)

def dict_combination(dict1,dict2,ratio = 0.5):
	output_dict = {}
	for one_proj in dict1:
		if one_proj in output_dict:
			output_dict[one_proj][0] = output_dict[one_proj][0] + ratio* dict1[one_proj][0]
		else:
			output_dict[one_proj] = [ratio* dict1[one_proj][0], dict1[one_proj][1]]

	for one_proj in dict2:
		if one_proj in output_dict:
			output_dict[one_proj][0] = output_dict[one_proj][0] + (1-ratio)* dict2[one_proj][0]
		else:
			output_dict[one_proj] = [(1-ratio)* dict2[one_proj][0], dict2[one_proj][1]]

def qlm_qe(params):

	index_dir = params.index_dir
	dataset_name = params.dataset_name
	base_model = params.base_model
	include_interaction = params.include_interaction
	expansion_type = params.expansion_type
	window_size = params.window_size
	max_dependency_count = params.max_dependency_count
	original_expansion_ratio = params.original_expansion_ratio
	current_history_ratio = params.current_history_ratio
	initial_list_size = params.initial_list_size
	reranking_list_size = params.reranking_list_size
	top_term_num = params.top_term_num
	expanded_vocabulary_size = params.expanded_vocabulary_size
	dirichlet_mu = params.dirichlet_mu
	parameter_str = '_'.join([base_model, str(include_interaction),expansion_type,str(window_size),str(max_dependency_count),str(original_expansion_ratio),str(current_history_ratio),str(initial_list_size),str(reranking_list_size),str(top_term_num),str(expanded_vocabulary_size),str(dirichlet_mu)])
	session_dir = os.path.join('sessiontrack',dataset_name, parameter_str)
	if not os.path.exists(session_dir):
		os.mkdir(session_dir)
	indri_ranking_path = os.path.join(session_dir, 'indri_ranking.txt')
	initial_ranking_path = os.path.join(session_dir,'initial_ranking.txt')
	reranking_path = os.path.join(session_dir,'reranking.txt')
	xml_file_path = os.path.join('sessiontrack',dataset_name,'sessiontrack'+dataset_name+'.xml')


	indri = IndriAPI(index_dir)
	session_list = parse_query_log(xml_file_path)
	gqlm = GQLM()
	indri_ranking_file = codecs.open(indri_ranking_path,'w')
	initial_ranking_file = codecs.open(initial_ranking_path,'w')
	reranking_file = codecs.open(reranking_path, 'w')


	for session_id in session_list:
		print("Processing session {}".format(session_id))


		#get current query
		query_text = session_list[session_id][0]
		query_text = preprocess(query_text)

		current_query = indri.get_query_array(query_text)

		# get the query vocabulary
		term_list = indri.get_query_array(query_text)

		interaction_query_list = []
		interaction_term_list = []

		# If interaction is considered, add interaction query terms to the term list
		if include_interaction:
			interaction_list = session_list[session_id][1]
			interaction_query_list, interaction_term_list = extraction_session_history(indri, interaction_list)
			for term in interaction_term_list:
				if term not in term_list:
					term_list.append(term)

		# get collection LM for Dirichlet smoothing
		collection_counts = [indri.get_collection_term_count(term) for term in term_list]
		collection_lm = np.diag(collection_counts)
		collection_lm = collection_lm/np.trace(collection_lm)


		# get idf vector for term list
		query_array = get_array(indri, term_list)

		#generate all possible dependencies up to a maximum number of terms involved
		query_dependency_list = generate_dependency_list(term_list, query_array, max_dependency_count)


		query_projector_dict = extract_quantum_events(indri,query_dependency_list, [current_query], window_size)

		# If interaction is considered, linearly combine current quantum events with historical words
		if include_interaction:
			history_projector_dict = extract_quantum_events(indri,query_dependency_list, interaction_query_list, window_size)
			query_projector_dict = dict_combination(query_projector_dict, history_projector_dict,current_history_ratio)

		# Train the query density matrix
		gqlm.prepare(query_projector_dict)
		gqlm.train()
		query_density_matrix = gqlm.rhoM



		# Generate Indri ranking and select the top k documents
		print("Write indri query result to file:")

		doc_list = indri.get_indri_ranking(query_text, initial_list_size)
		indri_doc_score_dic = {}
		for document_id in doc_list:
			indri_doc_score_dic [document_id] = doc_list[document_id]

		# write to file and return sorted document scores
		sorted_doc_list = write_result_to_file(indri_doc_score_dic, session_id,indri_ranking_file)


		# First GQLM
		initial_doc_score_dic = {}
		for doc_name in sorted_doc_list:

			# get document array
			doc_array = indri.get_document_array(doc_name)

			# get document quantum events
			projector_dict = extract_quantum_events(indri,query_dependency_list, [doc_array], window_size)

			# train document density matrix
			if(len(projector_dict) > 0):
				gqlm.prepare(projector_dict)
				gqlm.train()

			doc_density_matrix = gqlm.rhoM

			# get number of projectors for this document
			projector_count = get_quantum_event_count(indri,query_dependency_list, [doc_array], window_size)

			# dirichlet smoothing
			alpha_d =  dirichlet_mu/(projector_count + dirichlet_mu)
			doc_density_matrix = (1-alpha_d)* doc_density_matrix + alpha_d*collection_lm

			# vn_divergence as scores
			score = vn_divergence(doc_density_matrix, query_density_matrix)
			initial_doc_score_dic[doc_name] = np.real(score)

		print('Write first QLM result to file:')
		initial_sorted_doc_list = write_result_to_file(initial_doc_score_dic, session_id,initial_ranking_file)

		# get topK returned documents
		topK_result = [initial_sorted_doc_list[i] for i in range(reranking_list_size)]
		topK_result_str = " ".join(topK_result)

		# get tfidf dic for topK documents
		tf_idf = indri.get_tf_idf_4docs(topK_result_str)
		sorted_tf_idf = sorted(tf_idf, key=tf_idf.__getitem__, reverse = True)

		# take top terms plus original query terms
		num_terms = 0
		top_term_list = []
		# add original query terms
		for term in term_list:
			num_terms = num_terms+1
			top_term_list.append(term)

		# classical query expansion, directly taking top tfidf terms
		if expansion_type == 'classical':
			rereanking_term_list = term_list
			for i in range(len(sorted_tf_idf)):
				term = sorted_tf_idf[i]
				if not term in reranking_term_list:
					num_terms = num_terms+1
					reranking_term_list.append(term)
				if num_terms == expanded_vocabulary_size:
					break

		# quantum query expansion, building QLM for top documents
		elif expansion_type == 'quantum':
			for i in range(len(sorted_tf_idf)):
				term = sorted_tf_idf[i]
				if not term in top_term_list:
					num_terms = num_terms+1
					top_term_list.append(term)
				if num_terms == top_term_num:
					break


			# Training one single QLM for topK documents
			print("Training the QLM for topK result:")
			doc_array_list = []
			for one_doc_name in topK_result:
				doc_array_list.append(indri.get_document_array(one_doc_name))


			top_term_array = get_array(indri,top_term_list)
			dependency_list = generate_dependency_list(top_term_list, top_term_array, max_dependency_count)
			projector_dict = extract_quantum_events(indri,dependency_list, doc_array_list, window_size)
			gqlm.prepare(projector_dict)
			gqlm.train()

			#Sort the diagonal values to pick out the top K expansion terms
			indexes = np.argsort(np.diag(gqlm.rhoM))
			reranking_term_list = term_list
			num_terms = 0
			for i in range(len(indexes)):
				term = top_term_list[indexes[i]]
				if term not in top_term_list:
					num_terms = num_terms+1
					reranking_term_list.append(term)
				if num_terms == expanded_vocabulary_size:
					break

		# Conduct another QLM on the expanded term list
		# Compute collection LM for expanded term list
		reranking_collection_counts = [indri.get_collection_term_count(term) for term in reranking_term_list]
		collection_lm = np.diag(reranking_collection_counts)
		collection_lm = collection_lm/np.trace(collection_lm)

		# expansion_term_array = get_array(indri,expansion_term_list)
		reranking_term_array = get_array(indri, reranking_term_list)
		reranking_dependency_list = generate_dependency_list(reranking_term_list, reranking_term_array, max_dependency_count)

		# The projectors are extracted from the topK results
		reranking_projector_dict = extract_quantum_events(indri,reranking_dependency_list, doc_array_list, window_size)

		# Still linearly combine with history queries if history is considered
		if include_interaction:
			history_projector_dict = extract_quantum_events(indri,reranking_dependency_list, interaction_query_list, window_size)
			reranking_projector_dict = dict_combination(reranking_projector_dict, history_projector_dict,current_history_ratio)

		# Train the reranking density matrix
		gqlm.prepare(reranking_projector_dict)
		gqlm.train()
		reranking_query_density_matrix = gqlm.rhoM


		# Compute the query relevance score for each document
		reranking_doc_score_dic = {}
		for doc_name in sorted_doc_list:
			doc_array = indri.get_document_array(doc_name)

			projector_dict = extract_quantum_events(indri,reranking_dependency_list, [doc_array], window_size)
			if(len(projector_dict) > 0):
				gqlm.prepare(projector_dict)
				gqlm.train()
			doc_density_matrix = gqlm.rhoM

			projector_count = get_quantum_event_count(indri,reranking_dependency_list, [doc_array], window_size)
			alpha_d =  dirichlet_mu/(projector_count + dirichlet_mu)
			doc_density_matrix = (1-alpha_d)* doc_density_matrix + alpha_d*collection_lm

			score = vn_divergence(doc_density_matrix, reranking_query_density_matrix)
			reranking_doc_score_dic[doc_name] = score

		# Write reranking results to file
		print('Write reranking result to file:')
		reranking_sorted_doc_list = write_result_to_file(reranking_doc_score_dic, session_id, reranking_file)
		sorted_dic = sorted(reranking_doc_score_dic, key=reranking_doc_score_dic.__getitem__, reverse = True)

def main():
	params = Params()
	# params.parse_config('config/config_test.ini')
	# params.parseArgs()
	qlm_qe(params)





def get_array(indri,term_list):
	array = []
	for term in term_list:
		if indri.get_doc_frequency(term) == 0:
			array.append(2)
		else:
			array.append(1/indri.get_doc_frequency(term))
	return array

def preprocess(query_text):
	text = query_text
	text = strip_punctuation(text)
	text = remove_stop_words(text)
	text = stem_words(text)
	return text


def stem_words(text):
	stemmer = PorterStemmer()
	words = text.split()
	new_text = ''
	for word in words:
		new_text = new_text+ stemmer.stem(word) +' '

	new_text = new_text.strip()
	return new_text

def stem_document(document_array):
	stemmer = PorterStemmer()
	new_array = []
	for word in document_array:
		if word =='[OOV]':
			new_array.append('[OOV]')
		else:
		 	new_array.append(stemmer.stem(word))

	return new_array

	# dictionary = pyndri.extract_dictionary(index)
	# session_dir = '../sessiontrack/2014'
	# xml_file_path = os.path.join(session_dir,'sessiontrack2014.xml')
	# session_list = parse_query_log(xml_file_path)
	# num_results = 20
	# ranking_file_path = os.path.join(session_dir,'ranking.txt')
	# ranking_file = codecs.open(ranking_file_path,'w')
	# for session_id in session_list:
	# 	query_text = session_list[session_id]
	# 	query_text = strip_punctuation(query_text)
	# 	query_text = remove_stop_words(query_text)
	# 	print(query_text)
	# 	rank = 1
	# 	results = index.query(query_text,results_requested=num_results)
	# 	for int_document_id, score in results:
	# 		ext_document_id, _= index.document(int_document_id)
	# 		line = session_id+'\t'+'Q0'+'\t'+ext_document_id+ '\t'+ str(rank)+ '\t' + str(score) + '\t'+ 'Exp'+'\n'
	# 		rank = rank+1
	# 		ranking_file.write(line)


	# print(results)
def eval_dir(dir_path, qrel_path = 'sessiontrack/2014/qrel_per_session.txt'):
	run_path_original = os.path.join(dir_path,'indri_ranking.txt')
	run_path_reranking = os.path.join(dir_path, 'reranking.txt')
	qrels = QRels(qrel_path)
	run_reranking = TrecRun(run_path_reranking)
	# print(evaluate(run_origin, qrels, measures=[meanAvgPrecAt(10), ndcgAt(1),ndcgAt(5), ndcgAt(10)]))
	print(evaluate(run_reranking, qrels, measures=[meanAvgPrecAt(10), ndcgAt(10)]))



def create_session_qrel(session_dir, index):
	topic_qrel_path = os.path.join(session_dir, 'judgements.txt')
	with codecs.open(topic_qrel_path) as topic_qrel_file:
		topic_qrel_lines = topic_qrel_file.readlines()


	session_topic_mapping_path = os.path.join(session_dir, 'session-topic-mapping.txt')
	session_qrel_lines = []
	with codecs.open(session_topic_mapping_path) as session_topic_mapping_file:
		for line in session_topic_mapping_file.readlines():
	 		sessionID = line.split()[0]
	 		topicID = line.split()[1]
	 		for line in topic_qrel_lines:
	 			if(line.split()[0] == topicID):
	 				if(len(index.document_ids([line.split()[2]]))>=0):
	 					newline = sessionID + '\t'+ line.split()[1]+'\t'+line.split()[2]+'\t'+line.split()[3]
	 					session_qrel_lines.append(newline+'\n')
	 				# else:
	 					# print('This file does not exit in the index: {}'.format(line.split()[2]))
	output_file_path = os.path.join(session_dir, 'qrel_per_session.txt')
	with codecs.open(output_file_path,'w') as output_file:
		output_file.writelines(session_qrel_lines)

def parse_query_log(xml_file_path):
	tree = ET.parse(xml_file_path)
	root = tree.getroot()
	session_list = {}
	for session_node in root:
		sessionID = session_node.attrib['num']
		if int(sessionID)>100:
			break

		if session_node.find('currentquery') is None:
			continue

		query_text = session_node.find('currentquery').find('query').text
		historical_query_list = []
		for interaction_node in session_node.findall('interaction'):
			historical_query_list.append(interaction_node.find('query').text)

		session_list[sessionID] = (query_text, historical_query_list)
	return session_list

def is_stop_word(term):
	stop_words = set(stopwords.words('english'))
	if term in stop_words:
		return(True)
	else:
		return(False)

def remove_stop_words(text):
	stop_words = set(stopwords.words('english'))
	words = text.split()
	new_text = ''
	for word in words:
		if not word in stop_words:
			new_text = new_text+ word +' '

	new_text = new_text.strip()
	return new_text


def strip_punctuation(text):
    return ''.join(c for c in text if c not in string.punctuation)

if __name__ == '__main__':
	main()



