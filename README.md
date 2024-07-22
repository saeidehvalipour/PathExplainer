# PathExplainer

PathExplainer is a Python-based framework for evaluating and explaining paths in scientific literature using large language models (LLMs). This repository contains modules for evaluating LLM responses and iteratively improving them through a feedback loop.

## Features

- **AgathaEvaluator**: Evaluates LLM responses using a predefined knowledge base Agatha model.
- **HGCR**: Used as Retrieval component to retrieve paths from co-occurrence graph between pairs.
- **FeedbackLoop**: Iteratively refines LLM responses to achieve better explanations of indirect relationships between entities/pairs.
