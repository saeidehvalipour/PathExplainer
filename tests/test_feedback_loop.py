import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from feedback_loop import FeedbackLoop
from agatha_evaluator import AgathaEvaluator

class TestFeedbackLoop(unittest.TestCase):
    def setUp(self):
        # Mock AgathaEvaluator
        self.mock_evaluator = MagicMock(spec=AgathaEvaluator)
        
        # Mock model and pred_cui_pairs_set
        self.mock_model = MagicMock()
        self.mock_pred_cui_pairs_set = set()

        # Mock sents_db
        self.mock_sents_db = MagicMock()

        # Initialize the FeedbackLoop with mocked dependencies
        self.feedback_loop = FeedbackLoop(
            model=self.mock_model,
            pred_cui_pairs_set=self.mock_pred_cui_pairs_set,
            sents_db=self.mock_sents_db,
            evaluator=self.mock_evaluator
        )

    @patch('feedback_loop.retrieve_abstracts_from_pmids')
    @patch('feedback_loop.select_initial_pmids')
    @patch('feedback_loop.update_pmids')
    @patch('feedback_loop.find_top_documents_for_queries')
    @patch('feedback_loop.oai_get_response')
    def test_feedback_loop_all_paths(self, mock_oai_get_response, mock_find_top_documents_for_queries, mock_update_pmids, mock_select_initial_pmids, mock_retrieve_abstracts_from_pmids):
        # Define the test input DataFrame
        data = {
            'context_pmids': [['pmid1', 'pmid2'], ['pmid3', 'pmid4']]
        }
        top_20_df = pd.DataFrame(data)

        # Mock the return values of the dependencies
        mock_select_initial_pmids.return_value = ['pmid1', 'pmid2']
        mock_retrieve_abstracts_from_pmids.return_value = ['abstract1', 'abstract2']
        mock_oai_get_response.return_value = "llm_response"
        self.mock_evaluator.evaluate_llm_resp.return_value = [{'score': 0.9, 'sent_text': 'text', 'status': 'REWORK'}]
        mock_find_top_documents_for_queries.return_value = [['pmid3']]
        mock_update_pmids.return_value = ['pmid3']

        # Call the method under test
        result_df = self.feedback_loop.feedback_loop_all_paths(top_20_df)

        # Assert the results
        self.assertEqual(len(result_df), 1)
        self.assertIn('Path Index', result_df.columns)
        self.assertIn('Iteration Explanation', result_df.columns)
        self.assertIn('Agatha Score', result_df.columns)

        # Check if mocks were called with expected arguments
        mock_select_initial_pmids.assert_called_once()
        mock_retrieve_abstracts_from_pmids.assert_called()
        mock_oai_get_response.assert_called()
        self.mock_evaluator.evaluate_llm_resp.assert_called()
        mock_find_top_documents_for_queries.assert_called()
        mock_update_pmids.assert_called()

    def test_feedback_loop_no_rework(self):
        # Define the test input DataFrame
        data = {
            'context_pmids': [['pmid1', 'pmid2']]
        }
        top_20_df = pd.DataFrame(data)
        self.mock_evaluator.evaluate_llm_resp.return_value = [{'score': 0.9, 'sent_text': 'text', 'status': 'ok'}]
        result_df = self.feedback_loop.feedback_loop_all_paths(top_20_df)


        self.assertEqual(len(result_df), 1)
        self.assertIn('Path Index', result_df.columns)
        self.assertIn('Iteration Explanation', result_df.columns)
        self.assertIn('Agatha Score', result_df.columns)

    def test_feedback_loop_max_iterations(self):
        # Define the test input DataFrame
        data = {
            'context_pmids': [['pmid1', 'pmid2']]
        }
        top_20_df = pd.DataFrame(data)
        self.mock_evaluator.evaluate_llm_resp.return_value = [{'score': 0.9, 'sent_text': 'text', 'status': 'REWORK'}]
        self.feedback_loop.max_iterations = 1

        result_df = self.feedback_loop.feedback_loop_all_paths(top_20_df)

        # Assert the results
        self.assertEqual(len(result_df), 1)
        self.assertIn('Path Index', result_df.columns)
        self.assertIn('Iteration Explanation', result_df.columns)
        self.assertIn('Agatha Score', result_df.columns)

if __name__ == '__main__':
    unittest.main()
