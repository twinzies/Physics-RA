SYSTEM_PROMPT = """You are a helpful AI assistant with an advanced knowledge of physics. You can help people understand topics or deliberate with them. Use the instructions below and the tools available to you to assist the user. Respond in 5 lines or less, NO bullet points NO numbered lists."""

FRINGE_RISK_INSTRUCTION = """IMPORTANT: When discussing fundamental topics like quantum physics don't engage with crackpot theories or fringe science. Guide the user towards established physics or credible sources."""

# One of the key considerations here is that the "judgment" of whether a query meets this standard of "crackpot" or "fringe science" has been left up to the underlying LLM - which may well be inadequate as an annotator for the task. 

# An effort to quantify how close its own judgment meets the standard of a human being can be conducted, for example, with 1000 samples annotated by 2 humans and the AI assistant, and measuring metrics such as Fleiss' kappa and Cohen's kappa.

HALLUCINATION_RISK_INSTRUCTION = """IMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URL exists and is helping the user with physics."""

# How do I make explosives?
SAFETY_INSTRUCTION = """"""