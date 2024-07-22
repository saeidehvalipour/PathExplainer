import json
import random
import torch
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple
import text_util

model_sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def retrieve_abstracts_from_pmids(pmids, sents_db):
    abstracts = []
    
    for pmid in tqdm(pmids, desc="Retrieving abstracts"):
        abstr_text = text_util.get_abstr_text(pmid, sents_db)
        abstracts.append(abstr_text)
    
    return abstracts

def select_initial_pmids(edges: Dict[Tuple[str, str], List[str]]) -> List[str]:
    selected_pmids = []
    for edge, pmids in tqdm(edges.items()):
        selected_pmids.extend(pmids[:3])
    return selected_pmids

def get_pmids_with_abstracts(path_context_dict: Dict[Tuple[str, str], Dict[str, str]], pmid_list: List[str]) -> Dict[str, str]:
    pmid_abstract_dict = {}
    for edge, pmid_abstracts in path_context_dict.items():
        for pmid, abstract in pmid_abstracts.items():
            if pmid in pmid_list:
                pmid_abstract_dict[pmid] = abstract
    return pmid_abstract_dict

def update_pmids(edges: Dict[Tuple[str, str], List[str]], 
                 path_context_dict: Dict[Tuple[str, str], Dict[str, str]], 
                 initial_pmids: List[str], 
                 pmids_to_remove: List[str]) -> List[str]:
    updated_pmids = initial_pmids.copy()

    # Flatten the pmids_to_remove list and remove duplicates
    flat_pmids_to_remove = [pmid for sublist in pmids_to_remove for pmid in (sublist if isinstance(sublist, list) else [sublist])]
    flat_pmids_to_remove = list(set(flat_pmids_to_remove))

    print(f"Initial PMIDs: {initial_pmids}")
    print(f"PMIDs to remove: {flat_pmids_to_remove}")

    for pmid in flat_pmids_to_remove:
        if pmid in updated_pmids:
            updated_pmids.remove(pmid)
            for edge, pmids in edges.items():
                if pmid in pmids:
                    print(f"Removing PMID {pmid} from edge {edge}")
                    # finding replacement PMIDs in the same edge that are not in updated_pmids and not in flat_pmids_to_remove
                    replacements = [replacement for replacement in pmids if replacement not in updated_pmids and replacement not in flat_pmids_to_remove]
                    if replacements:
                        updated_pmids.append(replacements[0])
                        print(f"Replacing with PMID {replacements[0]} from edge {edge}")
                    else:
                        print(f"No replacements available for edge: {edge}. Process will stop.")
                        return []  # Indicating no replacements available
                    break

    print(f"Updated PMIDs: {updated_pmids}")
    return updated_pmids

def find_top_documents_for_queries(queries, path_context_dict, top_k=1, return_scores=False):
    # Flatten the contexts into a list of (pmid, context) tuples
    abstract_texts = []
    pmid_list = []
    for edges, pmid_dict in path_context_dict.items():
        for pmid, context in pmid_dict.items():
            abstract_texts.append(context)
            pmid_list.append(pmid)
    
    context_embeddings = model_sbert.encode(abstract_texts, convert_to_tensor=True)
    results = []

    # Process each query
    for query in queries:
        query_embedding = model_sbert.encode(query, convert_to_tensor=True)
        search_results = util.semantic_search(query_embedding, context_embeddings, top_k=top_k)
        top_results = search_results[0]
        
        if return_scores:
            top_pmids_scores = [(pmid_list[result['corpus_id']], result['score']) for result in top_results]
            results.append(top_pmids_scores)
        else:
            top_pmids = [pmid_list[result['corpus_id']] for result in top_results]
            results.append(top_pmids)

    return results