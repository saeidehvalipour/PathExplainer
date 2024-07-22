import time
import pandas as pd
from tqdm import tqdm
import text_util
from openai_llm import oai_get_response
from utils import retrieve_abstracts_from_pmids, select_initial_pmids, update_pmids, find_top_documents_for_queries

#openai_llm
# - oai_get_response

#evaluator_utils
# - evaluate_llm_resp

class FeedbackLoop:
    def __init__(self, model, pred_cui_pairs_set, sents_db):
        self.model = model
        self.pred_cui_pairs_set = pred_cui_pairs_set
        self.sents_db = sents_db

    def feedback_loop_all_paths(self, top_20_df, max_iterations=3):
        all_iterations_explanations = []
        all_agatha_scores = []
        all_paths = []

        for idx, row in top_20_df.iterrows():
            print(f'\n-------------------------Processing Path {idx}-------------------------------------------\n')
            # Step 1: Get Path
            path_context_pmids = row['context_pmids']
            path_context_dict = text_util.get_path_context(path_context_pmids, self.sents_db)
            path_context_pmids_dict = {k: list(v.keys()) for k, v in path_context_dict.items()}  # Create the path_context_pmids_dict

            # Print the all pmids for path
            print(f"Here are all the PMIDs for path {idx}: {path_context_pmids_dict}")

            # Step 1: Select initial PMIDs
            initial_pmids = select_initial_pmids(path_context_pmids_dict)
            existing_context_pmids = set(initial_pmids)

            # Print the initial PMIDs that will be sent as context
            print(f"Here are the selected PMIDs for context: {initial_pmids}")

            # Retrieve abstracts for the initial PMIDs
            path_context_abstracts = retrieve_abstracts_from_pmids(initial_pmids, self.sents_db)

            print('\n------------------Step 2 Prepare prompt + context -----------------------------------\n')

            # Step 2: Prepare initial prompt
            source_cui = 'GABBR1'  # 'C1421501'
            target_cui = 'Baclofen'  # 'C0071097'
            llm_prompt_template = 'How would you describe an indirect relationship between {source} and {target} given the following scientific abstracts as context?'
            llm_fix_prompt_str = '\n\n'.join(path_context_abstracts)
            llm_fix_prompt_combo = llm_prompt_template.format(source=source_cui, target=target_cui) + '\n\n' + llm_fix_prompt_str
            print(llm_fix_prompt_combo)

            extra_run_due_to_no_new_pmids = False

            # Initialize lists to store iteration explanations and Agatha scores for this path
            iterations_explanations = []
            agatha_scores = []

            for iteration in range(max_iterations):
                print(f"*********************************************Iteration {iteration + 1} for Path {idx}**********************************")

                print('\n------------------Step 3: Response generated from LLM -----------------------------------\n')
                # Step 3: Get Response from LLM
                try:
                    llm_resp_path = oai_get_response(llm_fix_prompt_combo, temp=1e-19, top_p=1e-9, seed=1234)
                    print(f"Response from LLM at iteration {iteration + 1}:\n{llm_resp_path}")
                except Exception as e:
                    print(f"Error getting response from LLM at iteration {iteration + 1}: {e}")
                    time.sleep(20)  # Wait for 20 seconds before retrying
                    continue

                # Store the response for this iteration
                iterations_explanations.append(llm_resp_path)

                print('\n------------------Step 4: Start Evaluation --------------------------------------------------\n')
                # Step 4: Evaluate Response
                eval_result_path = evaluate_llm_resp(llm_resp_path)
                print('\n------------------Step 4: Evaluation by Agatha is Done -----------------------------------\n')

                # Extract the Agatha score from the evaluation results and store it
                if eval_result_path and all('score' in item for item in eval_result_path):
                    agatha_score = sum(item['score'] for item in eval_result_path) / len(eval_result_path)
                    agatha_scores.append(agatha_score)
                else:
                    print("Score key not found in evaluation results or eval_result_path is empty.")
                    agatha_scores.append(None)

                # Extract rework sentences
                rework_sentences = [item['sent_text'] for item in eval_result_path if item['status'] == 'REWORK']

                # Print rework sentences for the current iteration
                if rework_sentences:
                    print(f"Rework sentences at iteration {iteration + 1}: {rework_sentences}")
                else:
                    print(f"Termination Condition1: There is no single Rework Sentence found after Evaluation llm response, this is best response from llm achieved for path {idx}!!")
                    all_iterations_explanations.extend(iterations_explanations)
                    all_agatha_scores.extend(agatha_scores)
                    all_paths.extend([idx] * len(iterations_explanations))
                    break

                print('\n------------------Step 5 Update Context--------------------------------------------------\n')
                # Step 5: Update Context
                pmids_to_remove = []
                for rework_sent in rework_sentences:
                    top_pmids = find_top_documents_for_queries([rework_sent], path_context_dict, top_k=1)
                    print(f"Here is the top_1 closest pmid for this rework sentence from llm: {rework_sent}: {top_pmids}")
                    pmids_to_remove.extend(top_pmids)
                    for item in pmids_to_remove:
                        print(pmids_to_remove)

                # Update the PMIDs using the existing update_pmids function
                updated_pmids = update_pmids(path_context_pmids_dict, path_context_dict, list(existing_context_pmids), pmids_to_remove)
                print(f"Updated PMIDs after replacement : {updated_pmids}")

                # Check if new PMIDs are found to be replaced
                if not updated_pmids or set(updated_pmids) == existing_context_pmids:
                    if extra_run_due_to_no_new_pmids:
                        print(f"No new PMIDs found to replace the context for path {idx}. Ending loop after one extra run with updated prompt.")
                        all_iterations_explanations.extend(iterations_explanations)
                        all_agatha_scores.extend(agatha_scores)
                        all_paths.extend([idx] * len(iterations_explanations))
                        break

                    print(f"No new PMIDs found to replace the context for path {idx}. Running one extra time with updated prompt.")
                    extra_run_due_to_no_new_pmids = True

                    # Update the prompt to indicate incorrect sentence only once if there is no pmid to replace from path, drop path
                    llm_prompt_template = '''How would you describe an indirect relationship between {source} and {target} given the following scientific abstracts as context? 
                    
                    Please note that the following sentence from the last response is incorrect sentence: 
                    "{rework_sentence}"
                    
                    Context from scientific literature: 
                    '''
                    llm_fix_prompt_combo = llm_prompt_template.format(source=source_cui, target=target_cui, rework_sentence=rework_sentences[0]) + '\n\n' + llm_fix_prompt_str

                    continue

                extra_run_due_to_no_new_pmids = False  # Reset the flag if new PMIDs are found

                # Retrieve abstracts for new PMIDs
                new_context_abstracts = retrieve_abstracts_from_pmids(updated_pmids, self.sents_db)
                existing_context_pmids.update(updated_pmids)
                print(f"update existing context pmid : {existing_context_pmids}")

                # Log the new contexts
                print(f"New contexts at iteration {iteration + 1} for path {idx}: {new_context_abstracts}")

                # Add new contexts to the prompt
                if new_context_abstracts:
                    new_context_str = '\n\n'.join(new_context_abstracts)
                    llm_fix_prompt_combo = llm_prompt_template.format(source=source_cui, target=target_cui) + '\n\n' + new_context_str

                print('\n------------------Step 6 Update Prompt based on Memory of Rework Sentence-----------------------------------\n')
                # Prepare prompt for next iteration if needed
                llm_fix_prompt_combo = llm_prompt_template.format(source=source_cui, target=target_cui) + '\n\n' + new_context_str

            print(f"Termination Condition2: Maximum iterations reached for path {idx}, so this is the best response after iteratively improve llm response")
            all_iterations_explanations.extend(iterations_explanations)
            all_agatha_scores.extend(agatha_scores)
            all_paths.extend([idx] * len(iterations_explanations))

        # Combine all results into a DataFrame
        result_df = pd.DataFrame({'Path Index': all_paths, 'Iteration Explanation': all_iterations_explanations, 'Agatha Score': all_agatha_scores})
        return result_df