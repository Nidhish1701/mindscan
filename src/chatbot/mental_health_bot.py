"""
Mental Health Support Chatbot — Integrated with ML Classification.

A hybrid chatbot that combines:
    - Rule-based pattern matching for immediate mood detection
    - XGBoost ML model for clinical-grade 6-class classification
    - Normal/healthy state pre-filter to prevent misclassification
    - Empathetic, condition-specific responses with coping techniques

Supported conditions:
    1. Anxiety
    2. BPD (Borderline Personality Disorder)
    3. Bipolar Disorder
    4. Depression
    5. Mental Illness (general)
    6. Schizophrenia
    7. Normal (healthy/positive state)

This is NOT a replacement for professional help — it's a supportive companion.
"""

import re
import random
from typing import Dict, List, Optional


# ---------------------------------------------------------------
# NORMAL / POSITIVE STATE DETECTION
# ---------------------------------------------------------------

# Strong positive indicators — if these appear WITHOUT distress
# keywords, the user is in a normal/healthy state
POSITIVE_KEYWORDS = {
    'happy', 'great', 'wonderful', 'amazing', 'fantastic', 'excellent',
    'good', 'fine', 'well', 'better', 'blessed', 'grateful', 'thankful',
    'cheerful', 'joyful', 'content', 'satisfied', 'peaceful', 'calm',
    'relaxed', 'excited', 'thrilled', 'delighted', 'optimistic',
    'hopeful', 'positive', 'awesome', 'beautiful', 'love', 'enjoy',
    'fun', 'glad', 'proud', 'confident', 'strong', 'energetic',
    'motivated', 'inspired', 'refreshed', 'comfortable', 'secure',
    'stable', 'improving', 'recovered', 'healthy', 'vibrant',
}

POSITIVE_PHRASES = [
    r'\b(feeling\s+(good|great|better|fine|happy|wonderful|amazing|fantastic|okay|ok|alright))\b',
    r'\b(i\s+(am|feel|\'m)\s+(good|great|happy|fine|well|okay|ok|alright|better|wonderful|fantastic|blessed))\b',
    r'\b(things\s+are\s+(good|great|better|fine|improving|looking\s+up))\b',
    r'\b(i\s+love\s+(my\s+)?life)\b',
    r'\b(life\s+is\s+(good|great|beautiful|wonderful))\b',
    r'\b(today\s+(was|is)\s+(good|great|amazing|wonderful|fantastic|a\s+great\s+day))\b',
    r'\b(i\'?m\s+doing\s+(well|great|good|fine|okay|ok|alright))\b',
    r'\b(everything\s+is\s+(fine|good|great|okay|ok|alright|perfect))\b',
    r'\b(no\s+(complaints?|worries|problems?|issues?))\b',
    r'\b(couldn\'?t\s+be\s+(better|happier))\b',
    r'\b(on\s+top\s+of\s+the\s+world)\b',
    r'\b(having\s+a\s+(good|great|wonderful|nice|lovely)\s+(day|time|week))\b',
]

# Distress keywords — if ANY of these appear, do NOT classify as Normal
DISTRESS_INDICATORS = {
    'depress', 'suicid', 'hopeless', 'worthless', 'empty', 'numb',
    'anxious', 'panic', 'scared', 'terrif', 'dread', 'manic',
    'voices', 'hallucinat', 'paranoi', 'abandon', 'unstable',
    'cutting', 'self-harm', 'selfharm', 'overdose', 'kill',
    'die', 'death', 'hurt', 'pain', 'suffer', 'miserable',
    'crying', 'tears', 'breakdown', 'breakdown', 'stress',
    'overwhelm', 'burnout', 'exhaust', 'insomnia', 'nightmare',
    'lonely', 'isolated', 'alone', 'broken', 'shatter',
    'hate', 'angry', 'rage', 'furious', 'disgusted',
    'confused', 'lost', 'trapped', 'helpless', 'desperate',
    'bipolar', 'schizo', 'bpd', 'borderline', 'disorder',
    'mental illness', 'mental health', 'therapy', 'therapist',
    'medication', 'meds', 'diagnosis', 'diagnosed',
}


def _is_normal_state(text_lower: str) -> bool:
    """
    Determine if the user's message indicates a normal/healthy mental state.

    Uses a two-step approach:
        1. Check for positive keywords/phrases
        2. Ensure NO distress indicators are present
    """
    # Step 1: Check for positive signals
    has_positive = False

    # Check positive keywords
    words = set(re.findall(r'\b\w+\b', text_lower))
    positive_matches = words & POSITIVE_KEYWORDS
    if len(positive_matches) >= 1:
        has_positive = True

    # Check positive phrases
    if not has_positive:
        for pattern in POSITIVE_PHRASES:
            if re.search(pattern, text_lower):
                has_positive = True
                break

    if not has_positive:
        return False

    # Step 2: Ensure no distress indicators
    for indicator in DISTRESS_INDICATORS:
        if indicator in text_lower:
            return False

    return True


class MentalHealthChatbot:
    """Empathetic mental health support chatbot with ML integration."""

    def __init__(self):
        self.conversation_history = {}
        self._ml_model = None
        self._ml_tokenizer = None
        self._ml_label_encoder = None

        # ----------------------------------------------------------
        # PATTERN DEFINITIONS — all 6 clinical conditions + extras
        # ----------------------------------------------------------
        self.patterns = {
            # ======== CRISIS (highest priority) ========
            'crisis': {
                'patterns': [
                    r'\b(suicid|kill\s*my\s*self|end\s*(my|it)\s*(life|all)|want\s*to\s*die|self.?harm)(?:e|al|ing|s)?\b',
                    r'\b(no\s*reason\s*to\s*live|better\s*off\s*dead|can\'?t\s*go\s*on)(?:ing)?\b',
                    r'\b(goodbye\s*forever|final\s*note|not\s*worth\s*living|no\s*way\s*out)\b',
                    r'\b(cutting\s*(my\s*)?(self|wrist|arm)|overdos)(?:e|ing)?\b',
                ],
                'condition': 'Crisis',
                'responses': [
                    "I hear you, and I want you to know that you matter. What you're feeling right now is temporary, even though it doesn't feel that way. Please reach out to a crisis helpline — they're trained to help.\n\n📞 **iCall**: 9152987821\n📞 **Vandrevala Foundation**: 1860-2662-345\n🌐 **International**: findahelpline.com\n\nYou don't have to face this alone. 💙",
                    "Thank you for sharing something so personal. Your life has value, and there are people who care about you. Please consider talking to someone who can help right now.\n\n📞 **Crisis Helpline**: 9152987821\n\nI'm here to listen, but a trained counselor can provide the support you deserve. 💙",
                ],
            },

            # ======== 1. DEPRESSION ========
            'depression': {
                'patterns': [
                    r'\b(depress|hopeless|worthless|empty|numb|miserable)(?:ed|ion|ing|ness)?\b',
                    r'\b(no\s*(point|motivation|energy|purpose)|can\'?t\s*feel|don\'?t\s*care)\b',
                    r'\b(hate\s*my\s*(self|life)|everything\s*is\s*(bad|awful|terrible|meaningless))\b',
                    r'\b(nothing\s*matters|what\'?s\s*the\s*point|no\s*hope|lost\s*all\s*hope)\b',
                    r'\b(can\'?t\s*get\s*out\s*of\s*bed|don\'?t\s*want\s*to\s*(live|exist|wake\s*up))\b',
                    r'\b(life\s*is\s*(pointless|meaningless|not\s*worth))\b',
                ],
                'condition': 'Depression',
                'responses': [
                    "I'm sorry you're going through this. Depression can make everything feel impossible, but you've already shown courage by expressing how you feel. 💙\n\nHere's something that might help right now:\n🌱 **5-4-3-2-1 Grounding**: Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.\n\nSmall steps matter. You don't have to fix everything today.",
                    "What you're feeling is valid, and it's okay not to be okay. Depression lies to us — it tells us things won't get better, but they can and do. 💙\n\n💡 **Try this**: Set one tiny goal for today. Just one. Drink water, step outside for 2 minutes, or text someone you trust.\n\nYou're stronger than you think.",
                    "I hear you, and I want you to know you're not alone in this. Many people experience what you're describing, and there is help available. 💙\n\n🧘 **Box Breathing**: Breathe in 4 seconds, hold 4 seconds, out 4 seconds, hold 4 seconds. Repeat 4 times.\n\nThis moment will pass. Be gentle with yourself.",
                ],
            },

            # ======== 2. ANXIETY ========
            'anxiety': {
                'patterns': [
                    r'\b(anxi|panic|worried|scared|terrif|fearful|dread)(?:ety|ous|ed|ful|ing)?\b',
                    r'\b(can\'?t\s*(stop|breath|relax|sleep|calm\s*down)|racing\s*(thoughts?|mind|heart))\b',
                    r'\b(overwhelm|freak|nervous|on\s*edge|restless|tense|uneasy)(?:ed|ing)?\b',
                    r'\b(panic\s*attack|heart\s*(pounding|racing)|hyperventilat)(?:s|ing|ed)?\b',
                    r'\b(constant\s*worry|fear\s*of|afraid\s*of|scared\s*of|keep\s*worrying)\b',
                    r'\b(what\s*if|worst\s*case|catastroph|doom)(?:izing|ic|e)?\b',
                ],
                'condition': 'Anxiety',
                'responses': [
                    "Anxiety can feel overwhelming, but you're safe right now. Let's try to ground you. 💙\n\n🧘 **4-7-8 Breathing**:\n• Breathe IN through your nose for 4 seconds\n• HOLD for 7 seconds\n• Breathe OUT through your mouth for 8 seconds\n• Repeat 3-4 times\n\nYour body knows how to calm down — let's give it a chance.",
                    "I understand that feeling of your mind racing. You're not 'crazy' — anxiety is your brain's protective system in overdrive. 💙\n\n🌿 **Grounding exercise**: Feel your feet on the floor. Press them down firmly. Notice the sensation. You are HERE, you are NOW, and you are SAFE.\n\nTake it one breath at a time.",
                    "What you're experiencing sounds really difficult. Anxiety tricks us into thinking the worst will happen, but remember: anxiety is not reality. 💙\n\n📝 **Try writing down**: What am I worried about? What's the MOST likely outcome? What would I tell my best friend?\n\nYou've survived every anxious moment so far. You'll get through this one too.",
                ],
            },

            # ======== 3. BPD (Borderline Personality Disorder) ========
            'bpd': {
                'patterns': [
                    r'\b(bpd|borderline\s*personality)\b',
                    r'\b(fear\s*of\s*(abandon|being\s*left|losing\s*everyone|rejection))\b',
                    r'\b(everyone\s*(leaves|abandons|will\s*leave)|they\'?ll?\s*(leave|abandon))\b',
                    r'\b(identity\s*(crisis|confusion|issues?)|don\'?t\s*know\s*who\s*i\s*am)\b',
                    r'\b(emotional\s*(rollercoaster|instability|swings?)|emotions?\s*(are\s*)?(out\s*of\s*control|too\s*(much|intense)))\b',
                    r'\b(splitting|ideali[sz]|devalu)(?:ing|ation|e)?\b',
                    r'\b(intense\s*(relationships?|attachment|anger)|push\s*(people\s*)?(away|and\s*pull))\b',
                    r'\b(chronic\s*(emptiness|empty)|feeling\s*empty\s*inside|void\s*inside)\b',
                    r'\b(i\s*(hate|love)\s*(you|them)\s*(so\s*much)?.*\b(but|then)\b)\b',
                    r'\b(black\s*and\s*white\s*thinking|all\s*or\s*nothing)\b',
                ],
                'condition': 'BPD',
                'responses': [
                    "I hear you. Living with intense emotions and the fear of being abandoned is incredibly challenging. Your feelings are valid, even when they feel overwhelming. 💙\n\n🌊 **TIPP Skill** (for intense emotions):\n• **T**emperature: Hold ice or splash cold water on your face\n• **I**ntense exercise: Do jumping jacks for 60 seconds\n• **P**aced breathing: Inhale 4s, exhale 6s\n• **P**aired muscle relaxation: Tense and release each muscle group\n\nYour emotions are intense, but they don't define you.",
                    "It sounds like you're experiencing really intense feelings, and that's incredibly difficult. The fear of abandonment and emotional instability can be exhausting. 💙\n\n🧩 **Grounding for intense emotions**: Name the emotion you're feeling right now. Rate it 1-10. Watch it. It WILL change — emotions are temporary visitors, not permanent residents.\n\n💡 **DBT Tip**: Try 'opposite action' — if the urge is to push away, try gently reaching out instead.\n\nYou deserve stable, healthy connections.",
                    "What you're describing sounds like a really painful experience. The intensity of your emotions shows how deeply you feel — that's not a weakness, it's a part of you. 💙\n\n📝 **Try this**: Keep a brief emotion log — write down what triggered the feeling, the emotion itself, and its intensity. Over time, patterns emerge that can help you anticipate and manage.\n\n🤝 **Dialectical Behavior Therapy (DBT)** has helped many people with similar experiences. Consider reaching out to a DBT-trained therapist.\n\nYou are not 'too much.' You are enough.",
                ],
            },

            # ======== 4. BIPOLAR DISORDER ========
            'bipolar': {
                'patterns': [
                    r'\b(bipolar|manic|mania|hypomania)\b',
                    r'\b(mood\s*swings?|moods?\s*(keep\s*)?chang|emotional\s*rollercoaster)\b',
                    r'\b(high\s*(energy|euphoria)|feel\s*(invincible|unstoppable)|on\s*top\s*of\s*the\s*world)\b',
                    r'\b(can\'?t\s*sleep\s*(but|and)\s*(not|full\s*of)\s*energy|don\'?t\s*need\s*sleep)\b',
                    r'\b(rapid\s*thoughts?|spending\s*spree|impulsive|reckless)\b',
                    r'\b(extreme\s*(highs?\s*(and|then)\s*lows?|ups?\s*(and|then)\s*downs?|mood))\b',
                    r'\b(crash|crashed|crashing)\s*(after|from|hard)\b',
                    r'\b(feel\s*(great|amazing)\s*(one|1)\s*(day|moment).*\b(terrible|awful|low)\b)\b',
                    r'\b(grandiose|inflated\s*(ego|self)|god\s*complex)\b',
                    r'\b(cycle|cycling|episodes?)\s*(of\s*)?(depression|mania|mood)\b',
                ],
                'condition': 'Bipolar',
                'responses': [
                    "I hear you. Living with mood swings between highs and lows can feel exhausting and confusing. You're not alone in this. 💙\n\n📋 **Mood Tracking**: Try rating your mood from 1-10 three times a day (morning, afternoon, night). This helps identify patterns and triggers.\n\n⏰ **Routine is key**: Try to maintain consistent sleep/wake times, meal times, and daily structure — stability in routine helps stabilize mood.\n\nBoth the highs and lows are real. You deserve support through all of them.",
                    "What you're describing — the highs followed by crashes — sounds incredibly draining. Bipolar mood episodes are real, they're not your fault, and they can be managed. 💙\n\n🛡️ **During a high (manic) phase**:\n• Delay major decisions by 48 hours\n• Tell someone you trust how you're feeling\n• Avoid alcohol and caffeine\n• Stick to your sleep schedule\n\n🧘 **During a low phase**: Be gentle with yourself. You'll cycle through this.\n\nConsider working with a psychiatrist — mood stabilizers can be transformative.",
                    "The emotional extremes you're experiencing are valid and they're hard. Bipolar disorder is a medical condition, not a character flaw. 💙\n\n📝 **Create an 'Early Warning' list**: Write down the first signs that a manic or depressive episode is starting. Share this with someone you trust so they can help you notice.\n\n💊 If you're not already, medication (mood stabilizers) combined with therapy has the strongest evidence for managing bipolar disorder.\n\nYou are more than your episodes.",
                ],
            },

            # ======== 5. MENTAL ILLNESS (General) ========
            'mental_illness': {
                'patterns': [
                    r'\b(mental\s*(illness|health\s*(issue|problem|condition|struggle|disorder))|mentally\s*(ill|unwell|sick))\b',
                    r'\b(something\s*(is|\'s)\s*wrong\s*with\s*(me|my\s*head|my\s*mind|my\s*brain))\b',
                    r'\b(losing\s*my\s*mind|going\s*(crazy|insane|mad)|not\s*normal)\b',
                    r'\b(need\s*(help|therapy|a\s*therapist|counseling|professional\s*help))\b',
                    r'\b(struggling\s*(with|mentally)|can\'?t\s*function|barely\s*functioning)\b',
                    r'\b(emotional\s*(breakdown|wreck|mess)|falling\s*apart|coming\s*undone)\b',
                    r'\b(diagnosis|diagnosed|disorder|condition|treatment|medication|meds|therapy)\b',
                    r'\b(my\s*(mental\s*health|condition|disorder)\s*(is|has))\b',
                ],
                'condition': 'Mental Illness',
                'responses': [
                    "Thank you for being honest about what you're going through. Acknowledging that you're struggling takes real courage. 💙\n\n🏥 **First steps to getting help**:\n1. Talk to your primary care doctor — they can screen and refer you\n2. Look for a therapist on platforms like Practo, BetterHelp, or your local mental health center\n3. Call iCall (9152987821) for free professional guidance\n\n📝 Before an appointment, jot down: what you're feeling, when it started, and how it affects daily life.\n\nSeeking help is a sign of strength, not weakness.",
                    "I hear you, and I want you to know that struggling with mental health is more common than you think — you are not alone, and you are not broken. 💙\n\n🌱 **Things that can help right now**:\n• Establish a simple daily routine (sleep, meals, one activity)\n• Move your body for 15-20 minutes (even just walking)\n• Limit social media and news consumption\n• Talk to someone you trust\n\n💬 If you haven't already, consider professional support. A therapist can help you understand what you're experiencing and build coping strategies.\n\nYou deserve to feel better, and help is available.",
                    "What you're feeling is valid, and it's brave of you to talk about it. Mental health struggles don't make you weak or 'crazy' — they make you human. 💙\n\n🧠 **Remember**: Mental health conditions are medical conditions, just like diabetes or asthma. They're treatable, and people recover every day.\n\n📞 **Free/affordable resources in India**:\n• iCall: 9152987821\n• NIMHANS: 080-46110007\n• Vandrevala Foundation: 1860-2662-345\n\nTaking the first step is the hardest part — and you've already started by speaking up.",
                ],
            },

            # ======== 6. SCHIZOPHRENIA ========
            'schizophrenia': {
                'patterns': [
                    r'\b(schizo|psycho[st]|psychotic)\b',
                    r'\b(hear(ing)?\s*(voices?|things?|sounds?\s*that|people\s*talking))\b',
                    r'\b(voices?\s*(in\s*my\s*head|tell|said|saying|command))\b',
                    r'\b(see(ing)?\s*(things?\s*that\s*aren\'?t|shadows?|people\s*who\s*aren\'?t|halluc))\b',
                    r'\b(hallucin|delusion|paranoi)(?:at|al|ed|ing|ous|a)?\b',
                    r'\b(people\s*(are|\'re)\s*(watching|following|plotting|out\s*to\s*get|spying))\b',
                    r'\b(government|they)\s*(are|\'re)\s*(tracking|monitoring|controlling|watching)\b',
                    r'\b(thought\s*(insertion|broadcast|control|withdrawal))\b',
                    r'\b(reality\s*(is|feels?|seems?)\s*(fake|distorted|not\s*real|confusing))\b',
                    r'\b(can\'?t\s*(tell|distinguish)\s*(what\'?s?\s*)?real)\b',
                    r'\b(disorganized|confused\s*(thinking|thoughts?)|nothing\s*makes\s*sense)\b',
                ],
                'condition': 'Schizophrenia',
                'responses': [
                    "What you're experiencing sounds really frightening, and I want you to know that you're not alone. These experiences — whether they're voices, visions, or distorted perceptions — are symptoms of a treatable medical condition. 💙\n\n🏥 **Important**: Please reach out to a psychiatrist or mental health professional as soon as possible. Antipsychotic medications can significantly reduce these symptoms.\n\n🌿 **Right now**:\n• Remind yourself: 'These are symptoms, not reality'\n• Go to a safe, calm environment\n• Call someone you trust\n\n📞 **NIMHANS Helpline**: 080-46110007\n📞 **iCall**: 9152987821\n\nYou are not 'crazy.' You have a medical condition that can be treated.",
                    "I hear you, and I want you to know that what you're experiencing — the voices, the paranoia, the confusion — these are recognized medical symptoms that millions of people experience. You are NOT losing your mind. 💙\n\n⚕️ **Key facts**:\n• Schizophrenia is a brain condition, not a character flaw\n• It affects about 1% of the global population\n• With treatment, many people live full, meaningful lives\n\n🛡️ **Coping strategies**:\n• Keep a 'reality check' journal — write down what feels real and ask someone you trust to verify\n• Maintain daily routines\n• Avoid recreational drugs and excess caffeine\n\nPlease seek professional help. Treatment works, and you deserve support.",
                    "That sounds incredibly scary and disorienting. I want you to know that what you're going through is a medical condition — it's not your fault, and it can be treated effectively. 💙\n\n🧠 **Grounding when reality feels distorted**:\n• Touch something cold or textured — focus on the sensation\n• Look around and name 3 objects you can definitely see\n• Call someone you trust and ask them to describe what they see/hear\n\n📋 **Next steps**: If you haven't already, please schedule an appointment with a psychiatrist. Early treatment leads to better outcomes.\n\nYou are brave for sharing this. Help is available.",
                ],
            },

            # ======== Stress (additional pattern) ========
            'stress': {
                'patterns': [
                    r'\b(stress|burn\s*out|exhausted|too\s*much|can\'?t\s*cope)(?:ed|ful|ing)?\b',
                    r'\b(pressure|deadline|overwork|tired\s*of|fed\s*up)\b',
                ],
                'condition': 'Stress',
                'responses': [
                    "It sounds like you're carrying a heavy load. It's okay to acknowledge that things are tough right now. 💙\n\n🎯 **Priority Reset**: Write down everything on your mind, then circle ONLY the top 3. Let the rest wait.\n\n☕ Take a 5-minute break. You deserve it.",
                    "Burnout is real, and recognizing it is the first step. You can't pour from an empty cup. 💙\n\n🌊 **Progressive Muscle Relaxation**: Starting from your toes, tense each muscle group for 5 seconds, then release. Work your way up to your shoulders.\n\nRest is not laziness — it's maintenance.",
                ],
            },

            # ======== Loneliness (additional pattern) ========
            'lonely': {
                'patterns': [
                    r'\b(lonely|alone|isolat|no\s*(one|friends)|nobody\s*(cares|understands))(?:ed|ion|ing)?\b',
                    r'\b(invisible|forgotten|abandoned|disconnected)(?:ed)?\b',
                ],
                'condition': 'Loneliness',
                'responses': [
                    "Feeling alone can be incredibly painful. But reaching out, even here, shows that part of you is looking for connection — honor that. 💙\n\n👥 **Small connection idea**: Send a 'thinking of you' message to someone. Join an online community about something you enjoy. Or visit a local library or café just to be around people.\n\nYou are worthy of connection.",
                    "Loneliness doesn't mean you're unlovable — it means you're human. We all need connection. 💙\n\n🌱 **Try this**: Do one social thing today, no matter how small. Reply to a message, smile at a stranger, or join an online discussion.\n\nConnection starts with one small step.",
                ],
            },

            # ======== Greeting ========
            'greeting': {
                'patterns': [
                    r'^(hello|hi+|hey+|good\s*(morning|afternoon|evening)|greetings?|yo|sup)\s*[?!.]*$',
                    r'^(hello|hi+|hey+)\s*[!.]*$',
                ],
                'condition': 'Greeting',
                'responses': [
                    "Hey! 👋 Great to see you here. How's your day going?",
                    "Hello! 😊 I'm here if you want to chat. What's on your mind?",
                    "Hey there! What's up? Feel free to share whatever you'd like. 💙",
                    "Hi! How are you doing today?",
                ],
            },

            # ======== Gratitude ========
            'gratitude': {
                'patterns': [
                    r'\b(thank(s|\s*you)|appreciate|grateful|helpful)\b',
                ],
                'condition': 'Gratitude',
                'responses': [
                    "Of course! I'm always here. 😊 Anything else on your mind?",
                    "Happy to help! 💙 Don't hesitate to reach out anytime.",
                    "Absolutely! That's what I'm here for. How are you feeling right now?",
                ],
            },

            # ======== Small Talk ========
            'smalltalk': {
                'patterns': [
                    r'^how\s+are\s+you(\s+doing)?\s*[?!.]*$',
                    r'^what\'?s\s+up\s*[?!.]*$',
                    r'^(how\s+(is|are)\s+(your|the)\s+day)\s*[?!.]*$',
                    r'^(are\s+you\s+(real|a\s+bot|an\s+ai|human))\s*[?!.]*$',
                    r'^(what\s+can\s+you\s+do|who\s+are\s+you|what\s+are\s+you)\s*[?!.]*$',
                ],
                'condition': 'SmallTalk',
                'responses': [
                    "I'm doing well, thanks for asking! 😊 I'm here to listen and chat. What about you — how are things?",
                    "Pretty good! I'm an AI mental health support companion. I'm here to listen, chat, and help however I can. What's going on with you?",
                    "Thanks for asking! I'm always ready to chat. 💙 What's on your mind today?",
                ],
            },

            # ======== Casual Self Introduction ========
            'self_intro': {
                'patterns': [
                    r'\b(my\s+name\s+is|i\s+am\s+called|call\s+me|myself)\s+[a-z]+\b',
                    r'\b(i\'?m|i\s+am)\s+[a-z]+(\s+[a-z]+)?,?\s*(nice|good|how|what)\b',
                ],
                'condition': 'Greeting',
                'responses': [],  # handled by name detection
            },
        }

        # Normal/Positive responses (used when normal state detected)
        self.normal_responses = [
            "That's genuinely great to hear! 😊 What's been making things feel good?",
            "Love hearing that! What's been going on? 🌟",
            "That's really nice. What's the highlight of your day been?",
            "Happy to hear you're doing well! Anything fun happening?",
        ]

        # Casual fallback responses — friendly, natural, varied
        self.casual_responses = [
            "I hear you! What's been on your mind lately?",
            "That's interesting — tell me more about that.",
            "Got it! Anything specific you wanted to talk about?",
            "I'm here! What's going on?",
            "Thanks for sharing that. What else is on your mind?",
        ]

        # Self-worth / emotional pain responses — warm and specific
        self.self_worth_responses = [
            "That really hurts to hear, and I'm glad you said something. Feeling useless or like you're not enough is one of the most painful things a person can experience — but those feelings are not facts. You matter more than you realize. 💙\n\nWhat's been making you feel this way?",
            "I'm really sorry you're feeling that way. That kind of thought — feeling like a burden or like you don't measure up — can be exhausting to carry. You don't have to carry it alone. What's been going on? 💙",
            "Hey, I want you to know — feeling useless or worthless is a sign that you're going through something really hard, not that those things are true about you. What's been happening? 💙",
            "That sounds like a really painful place to be. Feeling like you're not enough can take over everything. Can you tell me more about what's been going on for you? 💙",
        ]

        # Default response for unmatched distress
        self.default_responses = [
            "That sounds really hard. I'm here — can you tell me more about what you're going through? 💙",
            "I hear you, and I'm glad you're talking about it. What's been happening? 💙",
            "Thank you for sharing that. It takes courage to open up. What's been going on? 💙",
            "I'm listening — what's on your mind? 💙",
        ]

    def set_ml_model(self, model, tokenizer, label_encoder):
        """Set the ML model for classification (called during app startup)."""
        self._ml_model = model
        self._ml_tokenizer = tokenizer
        self._ml_label_encoder = label_encoder

    def _classify_with_ml(self, text: str) -> Optional[Dict]:
        """
        Classify the user's message using the XGBoost ML model.

        Returns None if model is not loaded, otherwise returns:
            {condition, confidence, probabilities}
        """
        if self._ml_model is None or self._ml_tokenizer is None:
            return None

        try:
            import numpy as np
            X_vec = self._ml_tokenizer.transform([text])
            probs = self._ml_model.predict_proba(X_vec)
            classes = list(self._ml_label_encoder.classes_)

            top_idx = int(np.argmax(probs[0]))
            top_confidence = float(probs[0][top_idx])
            top_class = str(classes[top_idx])

            return {
                'condition': top_class,
                'confidence': top_confidence,
                'probabilities': {
                    str(classes[j]): float(probs[0][j])
                    for j in range(len(classes))
                },
            }
        except Exception as e:
            print(f"[Chatbot] ML classification error: {e}")
            return None

    def respond(self, message: str) -> Dict:
        """
        Generate a supportive response based on the user's message.

        Processing pipeline:
            1. Check for Name Introductions
            2. Check for Normal/positive state (pre-filter)
            3. Pattern match against all condition categories
            4. Run ML model classification
            5. Combine results for best assessment

        Returns:
            Dict with response, mood_detected, suggestions, resources,
            and mental_health_assessment.
        """
        message_lower = message.lower().strip()

        # ----------------------------------------------------------
        # STEP 1: Check for Name Introductions
        # ----------------------------------------------------------
        name_match = re.search(r'\b(?:my\s+name\s+is|call\s+me|i\s+am\s+called)\s+([a-z]+)\b', message_lower)
        if name_match:
            name = name_match.group(1).capitalize()
            return {
                'response': f"Nice to meet you, {name}! 😊 I'm MindScan Support. How are you feeling today?",
                'mood_detected': 'neutral',
                'suggestions': self._get_suggestions('neutral'),
                'resources': None,
                'mental_health_assessment': {
                    'detected_condition': 'Greeting',
                    'confidence': 1.0,
                    'is_normal': False,
                    'ml_prediction': None,
                    'ml_confidence': None,
                    'note': 'User introduced themselves.'
                }
            }

        # ----------------------------------------------------------
        # STEP 2: Check for Normal/Healthy state
        # ----------------------------------------------------------
        if _is_normal_state(message_lower):
            response = random.choice(self.normal_responses)

            # Still run ML if available, but label as Normal
            ml_result = self._classify_with_ml(message)

            assessment = {
                'detected_condition': 'Normal',
                'confidence': 1.0,
                'is_normal': True,
                'ml_prediction': ml_result['condition'] if ml_result else None,
                'ml_confidence': ml_result['confidence'] if ml_result else None,
                'note': 'Your message indicates a healthy mental state. No mental health concerns detected.',
            }

            return {
                'response': response,
                'mood_detected': 'normal',
                'suggestions': self._get_suggestions('normal'),
                'resources': None,
                'mental_health_assessment': assessment,
            }

        # ----------------------------------------------------------
        # STEP 2: Pattern matching against all conditions
        # ----------------------------------------------------------
        matched_category = None
        for category, config in self.patterns.items():
            for pattern in config['patterns']:
                if re.search(pattern, message_lower):
                    matched_category = category
                    break
            if matched_category:
                break

        # ----------------------------------------------------------
        # STEP 3: Check if message has any distress signals at all
        # before running the expensive ML model.
        # Short/casual messages with no distress keywords should NOT
        # be forced through a clinical classifier.
        # ----------------------------------------------------------
        DISTRESS_WORDS = {
            # Clinical terms
            'depress', 'suicid', 'hopeless', 'worthless', 'empty', 'numb',
            'anxious', 'panic', 'scared', 'manic', 'voices', 'hallucinat',
            'paranoi', 'abandon', 'unstable', 'cutting', 'self-harm', 'overdose',
            'kill', 'die', 'hurt', 'pain', 'suffer', 'miserable', 'crying',
            'breakdown', 'overwhelm', 'burnout', 'insomnia', 'lonely', 'isolated',
            'broken', 'furious', 'helpless', 'desperate', 'bipolar',
            'schizo', 'bpd', 'borderline', 'disorder', 'mental health', 'therapy',
            'medication', 'diagnosis', 'stressed', 'no point', "can't cope",
            'giving up', 'falling apart', 'nothing matters',
            # Self-worth / emotional pain
            'useless', 'worthless', 'failure', 'loser', 'burden', 'pathetic',
            'hate myself', 'hate my life', 'not good enough', 'no good',
            'disappoint', 'ashamed', 'shame', 'embarrass', 'humiliat',
            'nobody cares', 'nobody loves', 'no one cares', 'no one loves',
            'i am nothing', 'i am nobody', 'i am a failure', 'i am useless',
            'feel like nothing', 'feel like nobody', 'feel invisible',
            'want to give up', 'tired of everything', 'tired of living',
            'lost all hope', 'no hope', 'no reason', 'not worth it',
            'drained', 'exhausted', 'numb inside', 'feel dead inside',
            'trapped', 'stuck', 'cant get out', 'ruined', 'destroyed',
            'angry at myself', 'mad at myself', 'blame myself', 'my fault',
            'unhappy', 'sad', 'crying', 'tears', 'upset', 'heartbroken',
            'rage', 'anger', 'furious', 'lost', 'confused', 'dont know',
        }
        words_in_msg = set(re.findall(r'\b\w+\b', message_lower))
        has_distress = any(w in message_lower for w in DISTRESS_WORDS)
        is_short_casual = len(message.split()) <= 8 and not has_distress

        # ----------------------------------------------------------
        # STEP 4: ML classification — only run when distress is present
        # ----------------------------------------------------------
        ml_result = None
        if has_distress or (matched_category and matched_category not in ['greeting', 'gratitude', 'smalltalk', 'self_intro']):
            ml_result = self._classify_with_ml(message)

        # ----------------------------------------------------------
        # STEP 5: Determine final condition
        # ----------------------------------------------------------
        # SELF-WORTH patterns — detect before other categories
        SELF_WORTH_PHRASES = [
            r'\b(i\s+(am|feel|think\s+i\s+am|think\s+i\'?m)\s+(useless|worthless|a\s+failure|a\s+burden|nothing|nobody|pathetic|a\s+loser))\b',
            r'\b(i\'?m\s+(useless|worthless|a\s+failure|a\s+burden|nothing|nobody|pathetic|a\s+loser))\b',
            r'\b(feel(ing)?\s+(useless|worthless|like\s+a\s+failure|like\s+a\s+burden|like\s+nothing|invisible|like\s+nobody))\b',
            r'\b(useless\s+(person|human|being)|worthless\s+(person|human|being))\b',
            r'\b(not\s+good\s+enough|no\s+good|such\s+a\s+failure|hate\s+my(self|\s+life))\b',
            r'\b(nobody\s+cares|no\s+one\s+cares|nobody\s+(loves|likes)\s+me|no\s+one\s+loves\s+me)\b',
            r'\b(useless\s+to\s+(everyone|anyone|people|them|others))\b',
        ]
        is_self_worth = any(re.search(p, message_lower) for p in SELF_WORTH_PHRASES)

        # CRISIS and CONVERSATIONAL patterns override everything
        if matched_category in ['crisis', 'greeting', 'gratitude', 'smalltalk']:
            config = self.patterns[matched_category]
            response = random.choice(config['responses'])
            detected_condition = config['condition']
        elif is_self_worth:
            # Self-worth issue — respond with specific warmth, label as depression-adjacent
            detected_condition = 'Depression'
            response = random.choice(self.self_worth_responses)
        elif is_short_casual and not matched_category:
            # Pure casual chat — respond conversationally, no clinical assessment
            detected_condition = 'Normal'
            response = random.choice(self.casual_responses)
        elif ml_result and ml_result['confidence'] >= 0.40:
            # Only use ML model when it's confident AND distress is present
            detected_condition = ml_result['condition']
            condition_to_category = {
                'Anxiety': 'anxiety',
                'BPD': 'bpd',
                'Bipolar': 'bipolar',
                'Depression': 'depression',
                'Mental Illness': 'mental_illness',
                'Schizophrenia': 'schizophrenia',
            }
            cat_key = condition_to_category.get(detected_condition, None)
            if cat_key and cat_key in self.patterns:
                response = random.choice(self.patterns[cat_key]['responses'])
            else:
                response = random.choice(self.default_responses)
        elif matched_category:
            # Fallback to pattern matching for other conditions
            config = self.patterns[matched_category]
            response = random.choice(config['responses'])
            detected_condition = config['condition']
        elif has_distress:
            # Has distress words but no clear match — ask empathetically
            detected_condition = 'Unknown'
            response = random.choice(self.default_responses)
        else:
            # Truly unclassified — respond casually
            detected_condition = 'Normal'
            response = random.choice(self.casual_responses)

        # ----------------------------------------------------------
        # Build mood for suggestions
        # ----------------------------------------------------------
        condition_to_mood = {
            'Crisis': 'crisis',
            'Depression': 'depressed',
            'Anxiety': 'anxious',
            'BPD': 'bpd',
            'Bipolar': 'bipolar',
            'Mental Illness': 'mental_illness',
            'Schizophrenia': 'schizophrenia',
            'Stress': 'stressed',
            'Loneliness': 'lonely',
            'Normal': 'normal',
            'Greeting': 'neutral',
            'Gratitude': 'neutral',
            'SmallTalk': 'neutral',
            'Unknown': 'neutral',
        }
        mood = condition_to_mood.get(detected_condition, 'neutral')

        # Build suggestions
        suggestions = self._get_suggestions(mood)

        # Crisis resources
        resources = None
        if mood == 'crisis':
            resources = {
                "iCall (India)": "9152987821",
                "Vandrevala Foundation": "1860-2662-345",
                "International": "https://findahelpline.com",
            }

        # Build assessment — always return one so probability is always visible
        CONVERSATIONAL = {'Greeting', 'Gratitude', 'SmallTalk', 'Normal'}
        is_conversational = detected_condition in CONVERSATIONAL

        if is_conversational:
            # Lightweight assessment — no probability bar, just a clean status pill
            assessment = {
                'detected_condition': detected_condition,
                'confidence': None,   # no bar shown for casual chat
                'is_normal': detected_condition == 'Normal',
                'ml_prediction': None,
                'ml_confidence': None,
                'note': None,
            }
        else:
            is_pattern_match = matched_category is not None and matched_category in ['crisis']
            final_confidence = 1.0 if is_pattern_match else (ml_result['confidence'] if ml_result else None)
            assessment = {
                'detected_condition': detected_condition,
                'confidence': final_confidence,
                'is_normal': False,
                'ml_prediction': ml_result['condition'] if ml_result else None,
                'ml_confidence': ml_result['confidence'] if ml_result else None,
                'note': None,   # removed note — it was redundant and noisy
            }

        # Only return suggestions for clinical/distress moods
        clinical_moods = {'crisis', 'depressed', 'anxious', 'bpd', 'bipolar', 'mental_illness', 'schizophrenia', 'stressed', 'lonely'}
        final_suggestions = suggestions if mood in clinical_moods else []

        return {
            'response': response,
            'mood_detected': mood,
            'suggestions': final_suggestions,
            'resources': resources,
            'mental_health_assessment': assessment,
        }

    def _get_condition_note(self, condition: str) -> str:
        """Get a brief note about the detected condition."""
        notes = {
            'Depression': 'Indicators of depressive symptoms detected. Professional support is recommended if these persist.',
            'Anxiety': 'Anxiety-related indicators detected. Breathing exercises and professional support can help.',
            'BPD': 'Indicators consistent with Borderline Personality Disorder detected. DBT therapy is highly effective.',
            'Bipolar': 'Mood instability patterns detected. Mood stabilizers and professional support can be very helpful.',
            'Mental Illness': 'General mental health distress detected. Consider speaking with a mental health professional.',
            'Schizophrenia': 'Symptoms that may indicate a psychotic condition. Professional psychiatric evaluation is strongly recommended.',
            'Crisis': '⚠️ CRISIS DETECTED. Please contact emergency services or a crisis helpline immediately.',
            'Stress': 'Elevated stress levels detected. Consider stress management techniques and self-care.',
            'Loneliness': 'Social isolation indicators detected. Building connections, even small ones, can help.',
            'Normal': 'Your message indicates a healthy mental state. No mental health concerns detected.',
            'Greeting': 'No mental health indicators in this message.',
            'SmallTalk': 'Casual conversation — no mental health indicators.',
            'Gratitude': 'No mental health indicators in this message.',
            'Unknown': 'Could not confidently determine mental health state. Would you like to share more?',
        }
        return notes.get(condition, 'Assessment could not be determined.')

    def _get_suggestions(self, mood: str) -> List[str]:
        """Get contextual suggestions based on detected mood."""
        suggestions_map = {
            'crisis': [
                "Call a crisis helpline immediately",
                "Go to your nearest emergency room",
                "Tell someone you trust how you're feeling",
            ],
            'depressed': [
                "Try the 5-4-3-2-1 grounding technique",
                "Set one small achievable goal for today",
                "Reach out to someone you trust",
                "Consider speaking with a counselor",
            ],
            'anxious': [
                "Practice 4-7-8 breathing",
                "Try progressive muscle relaxation",
                "Write down your worries",
                "Limit caffeine intake",
            ],
            'bpd': [
                "Practice the TIPP skill for intense emotions",
                "Use opposite action when urges are strong",
                "Try journaling to track emotional patterns",
                "Consider Dialectical Behavior Therapy (DBT)",
            ],
            'bipolar': [
                "Track your mood 3 times daily (1-10 scale)",
                "Maintain a consistent sleep schedule",
                "Delay major decisions during mood episodes",
                "Work closely with a psychiatrist on medication",
            ],
            'mental_illness': [
                "Schedule an appointment with a mental health professional",
                "Start a simple daily routine",
                "Talk to someone you trust about how you feel",
                "Explore free helplines like iCall (9152987821)",
            ],
            'schizophrenia': [
                "See a psychiatrist for proper evaluation",
                "Keep a reality-check journal",
                "Maintain daily routines and structure",
                "Avoid recreational drugs and excess stimulants",
            ],
            'stressed': [
                "Take a 5-minute break",
                "Prioritize your top 3 tasks only",
                "Try progressive muscle relaxation",
                "Get some fresh air",
            ],
            'lonely': [
                "Reach out to one person today",
                "Join an online community",
                "Consider volunteering",
                "Spend time in a public place like a café",
            ],
            'normal': [
                "Write down 3 things you're grateful for",
                "Share your positivity with someone",
                "Plan something to look forward to",
                "Keep up your healthy habits!",
            ],
            'positive': [
                "Write down 3 things you're grateful for",
                "Share your positivity with someone",
                "Plan something to look forward to",
            ],
        }
        return suggestions_map.get(mood, ["Take a deep breath", "Be kind to yourself today"])
