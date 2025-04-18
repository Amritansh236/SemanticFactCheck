# SemanticFactCheck
Flask-based web app that fact-checks user-submitted claims using real-time Google search and natural language inference (NLI) models.

## Features
- Searches the web for related articles
- Uses `roberta-large-mnli` for semantic entailment
- Determines if a claim is likely true or false
- Generates corrections with `flan-t5-large`
- Detects countries and platforms where the claim spread

## How it works
1. Enter a claim (e.g., "5G causes COVID-19")
2. App fetches articles via Google Custom Search
3. Each article is checked using NLI for entailment/contradiction
4. Verdict is calculated + a correction is generated
