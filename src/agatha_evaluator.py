import json
import random
import torch
from pathlib import Path
from tqdm import tqdm
from agatha.construct.semrep_handler import SemRepHandler

class AgathaEvaluator:
    def __init__(self, model_path, embedding_path, entity_db, graph_db):
        # Initialize the sr_h object
        nlm_soft_folder = '/lustre/acslab/users/3281/SemRep'
        sr_temp_folder = '/lustre/acslab/users/3281/semrep_temp_2024'
        sr_replace_utf8_path = '/lustre/acslab/users/3281/SemRep/replace_utf8.jar'

        self.sr_h = SemRepHandler(
            nlm_soft_path=nlm_soft_folder,
            temp_folder=sr_temp_folder,
            replace_utf8_path=sr_replace_utf8_path,
        )

        self.sr_h.sr_binary_path = Path('/lustre/acslab/users/3281/SemRep/public_semrep/bin/semrep.v1.9_2021AB')

        # Load the st_nodelbl_dict
        with open(
            '/lustre/acslab/shared/Agatha_shared/2021_11_22/2021_11_22_semtypes_to_nodelbl_and_vv_dict.json', 'r'
        ) as f:
            self.st_nodelbl_dict = json.load(f)

        # Initialize model and predicate CUI pairs
        self.model_path = model_path
        self.embedding_path = embedding_path
        self.entity_db = entity_db
        self.graph_db = graph_db

        self.model = self.load_model()
        self.pred_cui_pairs_set = self.extract_pred_cui_pairs()

    def load_model(self):
        model = torch.load(self.model_path)
        model.configure_paths(
            embedding_dir=self.embedding_path,
            entity_db=self.entity_db,
            graph_db=self.graph_db,
        )
        return model.eval()

    def extract_pred_cui_pairs(self):
        nodelist = self.model.graph.keys()
        pred_pairs_list = [p for p in tqdm(nodelist) if p[0] == 'p']

        pred_cui_pairs_set = set()

        for p in tqdm(pred_pairs_list):
            p_split = p.upper().split(':')
            s = p_split[1]
            o = p_split[-1]
            pred_cui_pairs_set.add(tuple(sorted([s, o])))

        return pred_cui_pairs_set

    def generate_negatives(self, pos_pair, sample_rate=5):
        s = pos_pair[0]
        o = pos_pair[1]

        if s[0] != 'm':
            s = f'm:{pos_pair[0].lower()}'

        if o[0] != 'm':
            o = f'm:{pos_pair[1].lower()}'

        if o in self.st_nodelbl_dict:
            o_st = self.st_nodelbl_dict[o]
            o_sample_list = random.sample(
                self.st_nodelbl_dict[o_st],
                sample_rate
            )
        else:
            o_sample_list = []

        out_pairs = []
        for o_neg in o_sample_list:
            out_pairs.append((s, o_neg))

        return out_pairs

    def eval_pair(self, pair, sample_rate=10):
        pair_negs = self.generate_negatives(pair, sample_rate=sample_rate)
        agatha_queries = [pair] + pair_negs
        pair_labels = [1] + [0] * len(pair_negs)
        scores = self.model.predict_from_terms(agatha_queries)

        res_list = sorted(
            list(zip(scores, pair_labels)),
            key=lambda x: x[0],
            reverse=True
        )

        rank = None
        for i, (score, lbl) in enumerate(res_list):
            if lbl == 1:
                rank = i
                break

        return {
            'scores': res_list,
            'pos_rank': rank + 1
        }

    def evaluate_llm_resp(self, llm_resp_str):
        expl_parts_list = []
        cleaned_text = llm_resp_str.replace('\n', '').replace('*', '')
        expl_sr_out = self.sr_h.ProcessList_parallel([cleaned_text])

        total_agatha_score = 0
        count_agatha_score = 0

        for s_id, sent_data in expl_sr_out.items():
            sent_text = sent_data['sent_text']
            status = 'ok'
            bad_predicates = []
            good_predicates = []
            agatha_score_path = []

            for rel in sent_data['relations']:
                rel_subj = rel['subj_text']
                v = rel['verb']
                rel_obj = rel['obj_text']

                pred = [rel_subj, rel_obj]
                rel_pair = tuple(
                    sorted(
                        [
                            rel['subj_id'],
                            rel['obj_id']
                        ]
                    )
                )
                if rel_pair in self.pred_cui_pairs_set:
                    good_predicates.append(pred)
                if rel_pair not in self.pred_cui_pairs_set:
                    subj_id = f"m:{rel['subj_id'].lower()}"
                    obj_id = f"m:{rel['obj_id'].lower()}"
                    if subj_id in self.model.graph and obj_id in self.model.graph:
                        agatha_score = self.model.predict_from_terms([[subj_id, obj_id]])[0]
                        agatha_rank = self.eval_pair([subj_id, obj_id], sample_rate=20)['pos_rank']
                        pred.append(agatha_rank)
                        if agatha_rank > 1:
                            status = 'REWORK'
                            bad_predicates.append(pred)
                            agatha_score_path.append(agatha_score)
                            print("Agatha score for predicate: {}".format(agatha_score))
                        else:
                            good_predicates.append(pred)
                            agatha_score_path.append(agatha_score)

                        total_agatha_score += agatha_score
                        count_agatha_score += 1

            print('Sentence:\n', sent_text)
            print('Status:\n', status)
            if status != 'ok':
                print('Reason:')
                for p in bad_predicates:
                    print('\t', p, 'not in AGATHA KB')
            else:
                if len(good_predicates):
                    print('Recognized:')
                for p in good_predicates:
                    print('\t', p, 'in AGATHA KB')

            print('\n-----\n')

            expl_parts_list.append(
                {
                    'sent_text': sent_text,
                    'status': status,
                    'bad_predicates': bad_predicates,
                    'good_predicates': good_predicates,
                    'agatha_score': agatha_score_path
                }
            )

        if count_agatha_score > 0:
            average_agatha_score = total_agatha_score / count_agatha_score
            print("Average Agatha score: {}".format(average_agatha_score))
        else:
            print("No Agatha scores available to calculate average.")

        return expl_parts_list
