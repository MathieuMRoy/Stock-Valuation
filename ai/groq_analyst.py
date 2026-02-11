"""
Groq AI Analyst - AI-powered stock analysis reports
"""

# Check for Groq availability
try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False


def ai_analyst_report(metrics: dict, bench: dict, scores: dict, tech: dict, api_key: str) -> tuple:
    """
    Generate an AI-powered analyst report using Groq.
    
    Args:
        metrics: Current stock metrics
        bench: Benchmark data
        scores: Scoring data
        tech: Technical analysis data
        api_key: Groq API key
    
    Returns:
        Tuple of (report_text, error_message)
    """
    if not GROQ_OK:
        return None, "Groq package missing."
    
    if not api_key:
        return None, "API Key missing."
    
    try:
        client = Groq(api_key=api_key)
        
        valuation_context = "Sous-évalué" if scores['valuation'] > 6 else "Sur-évalué"
        
        prompt = f"""
        Tu es Cameron Doerksen, Analyste Senior. Rédige un rapport de "Coverage" pour l'action {metrics['ticker']} (Données jan 2026).
        Prix: {metrics['price']}$. PE: {metrics['pe']}x. Croissance: {metrics['sales_gr']*100:.1f}%.
        Technique: {"Haussière" if tech['score'] > 6 else "Neutre"}.
        
        STRUCTURE DU RAPPORT:
        1. THÈSE D'INVESTISSEMENT (Note /10 et Recommandation: ACHAT/VENDRE)
        2. ANALYSE FONDAMENTALE (Forces/Faiblesses)
        3. VALORISATION ({valuation_context})
        4. STRATÉGIE TRADING (Prix d'entrée et Stop Loss précis)
        
        Ton analytique et professionnel. Français.
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content, None
    except Exception as e:
        return None, str(e)
