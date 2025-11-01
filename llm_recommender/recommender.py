# llm_recommender/recommender.py

def llm_recommender(prompt: str):
    """
    Mock version of the LLM recommender.
    Later, this will call the OpenAI API or Sentence-BERT for embeddings.
    """
    if not prompt:
        return {
            "prompt": "",
            "recommendations": [],
            "explanation": "No prompt received."
        }

    lower_prompt = prompt.lower()
    if "study" in lower_prompt or "focus" in lower_prompt:
        return {
            "prompt": prompt,
            "recommendations": ["Lo-Fi Beats", "Rainy Jazz", "Calm Piano"],
            "explanation": "Relaxing, instrumental tracks for concentration."
        }
    elif "run" in lower_prompt or "gym" in lower_prompt:
        return {
            "prompt": prompt,
            "recommendations": ["Power Up", "Adrenaline Rush", "Fast Tempo Mix"],
            "explanation": "High-energy songs ideal for running or workouts."
        }
    elif "sleep" in lower_prompt or "chill" in lower_prompt:
        return {
            "prompt": prompt,
            "recommendations": ["Deep Sleep Waves", "Dreamscape", "Evening Calm"],
            "explanation": "Soft ambient music for relaxation or rest."
        }
    else:
        return {
            "prompt": prompt,
            "recommendations": ["Daily Mix", "Morning Coffee", "Acoustic Essentials"],
            "explanation": "Balanced acoustic-electronic blend."
        }
