"""
Prompt templates for semantic query augmentation with Llama-3.1-8B-Instruct.
Engineered for video moment retrieval query generation.
"""

class PromptTemplateManager:
    """Manages prompts for different augmentation strategies."""
    
    SYSTEM_PROMPT = """You are an expert video annotation assistant specializing in creating semantically equivalent descriptions for video moments. Your task is to generate diverse variations of video moment queries while preserving exact semantic meaning."""
    
    TEMPLATES = {
        "semantic_paraphrase": """Generate {num_variations} semantically equivalent variations of the following video moment query.

Original Query: "{original_query}"

Video Context:
- Duration: {duration:.1f} seconds
- Moment Timespan: {start_time:.1f}s - {end_time:.1f}s

Requirements:
1. Maintain EXACT semantic meaning (same action, objects, timing implied)
2. Use DIFFERENT sentence structures and vocabulary
3. Keep queries concise (5-20 words)
4. Focus ONLY on observable visual content
5. Do NOT add information not in the original query
6. Do NOT change or remove key actions/objects

Output Format (JSON):
{{
  "variations": [
    "variation 1",
    "variation 2",
    ...
  ]
}}

Generate exactly {num_variations} variations as a valid JSON object.""",

        "synonym_replacement": """Rewrite the query using synonyms while preserving meaning.

Original Query: "{original_query}"

Generate {num_variations} variations by replacing key words with synonyms:
- Use natural synonyms (e.g., "person" → "individual", "open" → "push open")
- Maintain grammatical correctness
- Keep the same sentence structure when possible

Output Format (JSON):
{{
  "variations": [
    "variation 1",
    "variation 2"
  ]
}}""",

        "structure_variation": """Rewrite the query with different grammatical structures.

Original Query: "{original_query}"

Generate {num_variations} variations using:
1. Active/Passive voice changes
2. Verb-first vs Noun-first structures
3. Different verb tenses (if applicable)
4. Clause reordering

Keep meaning identical.

Output Format (JSON):
{{
  "variations": [
    "variation 1",
    "variation 2"
  ]
}}""",

        "mixed_strategy": """Generate diverse variations using multiple linguistic strategies.

Original Query: "{original_query}"

Video Context:
- Duration: {duration:.1f}s
- Moment: {start_time:.1f}s - {end_time:.1f}s

Apply these strategies:
1. Paraphrase with different vocabulary
2. Change sentence structure (active/passive)
3. Use synonyms for key terms
4. Vary detail level (more/less specific)

Generate {num_variations} high-quality variations that:
- Are semantically equivalent
- Sound natural and fluent
- Cover different linguistic patterns
- Maintain visual focus

Output Format (JSON):
{{
  "variations": [
    "variation 1",
    "variation 2",
    "variation 3"
  ]
}}"""
    }
    
    @staticmethod
    def get_prompt(strategy="semantic_paraphrase", **kwargs):
        """
        Get formatted prompt for a specific augmentation strategy.
        
        Args:
            strategy: One of the template keys
            **kwargs: Template variables (original_query, num_variations, etc.)
        
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        if strategy not in PromptTemplateManager.TEMPLATES:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(PromptTemplateManager.TEMPLATES.keys())}")
        
        # Set defaults
        kwargs.setdefault('num_variations', 5)
        kwargs.setdefault('duration', 150.0)
        kwargs.setdefault('start_time', 0.0)
        kwargs.setdefault('end_time', 10.0)
        
        user_prompt = PromptTemplateManager.TEMPLATES[strategy].format(**kwargs)
        
        return PromptTemplateManager.SYSTEM_PROMPT, user_prompt
    
    @staticmethod
    def get_all_strategies():
        """Return list of available strategies."""
        return list(PromptTemplateManager.TEMPLATES.keys())
