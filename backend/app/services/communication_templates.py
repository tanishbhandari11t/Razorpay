from __future__ import annotations

"""Deterministic communication templates used when Qwen is unavailable."""

from typing import Any

PAYMENT_LINK_PLACEHOLDER = "@@PAYMENT_LINK@@"

SUPPORTED_LANGUAGES = (
    "english",
    "hinglish",
    "hindi",
    "tamil",
    "telugu",
    "marathi",
    "bengali",
    "gujarati",
    "kannada",
    "malayalam",
    "punjabi",
)

LANGUAGE_LABELS: dict[str, str] = {
    "english": "English",
    "hinglish": "Hinglish",
    "hindi": "हिन्दी",
    "tamil": "தமிழ்",
    "telugu": "తెలుగు",
    "marathi": "मराठी",
    "bengali": "বাংলা",
    "gujarati": "ગુજરાતી",
    "kannada": "ಕನ್ನಡ",
    "malayalam": "മലയാളം",
    "punjabi": "ਪੰਜਾਬੀ",
}


TEMPLATES: dict[str, dict[str, str]] = {
    "payment_link": {
        "english": (
            "Hi {name},\n\n"
            "Your payment of {amount} couldn't be completed.\n\n"
            "You can securely complete your payment using the link below:\n"
            f"{PAYMENT_LINK_PLACEHOLDER}\n\n"
            "If you've already completed the payment, please ignore this message."
        ),
        "hinglish": (
            "Hi {name},\n\n"
            "Aapka {amount} payment complete nahi ho paya.\n\n"
            "Aap neeche diye gaye secure payment link se dobara complete kar sakte hain:\n"
            f"{PAYMENT_LINK_PLACEHOLDER}\n\n"
            "Agar aapne pehle hi payment kar diya hai, is message ko ignore kijiye."
        ),
        "hindi": (
            "नमस्ते {name},\n\n"
            "आपका {amount} भुगतान पूरा नहीं हो पाया।\n\n"
            "कृपया नीचे दिए सुरक्षित लिंक से भुगतान पूरा करें:\n"
            f"{PAYMENT_LINK_PLACEHOLDER}\n\n"
            "यदि आप पहले ही भुगतान कर चुके हैं, तो इस संदेश को अनदेखा करें।"
        ),
        "tamil": (
            "வணக்கம் {name},\n\n"
            "உங்கள் {amount} கட்டணம் முடிக்கப்படவில்லை.\n\n"
            "கீழே உள்ள பாதுகாப்பான இணைப்பைப் பயன்படுத்தி கட்டணத்தை முடிக்கவும்:\n"
            f"{PAYMENT_LINK_PLACEHOLDER}\n\n"
            "நீங்கள் ஏற்கனவே செலுத்தியிருந்தால் இந்த செய்தியை புறக்கணிக்கவும்."
        ),
        "telugu": (
            "నమస్కారం {name},\n\n"
            "మీ {amount} చెల్లింపు పూర్తి కాలేదు.\n\n"
            "కింది సురక్షిత లింక్ ద్వారా చెల్లింపు పూర్తి చేయండి:\n"
            f"{PAYMENT_LINK_PLACEHOLDER}\n\n"
            "మీరు ఇప్పటికే చెల్లించి ఉంటే ఈ సందేశాన్ని విస్మరించండి."
        ),
        "marathi": (
            "नमस्कार {name},\n\n"
            "तुमचे {amount} पेमेंट पूर्ण झाले नाही.\n\n"
            "कृपया खालील सुरक्षित लिंकने पेमेंट पूर्ण करा:\n"
            f"{PAYMENT_LINK_PLACEHOLDER}\n\n"
            "जर तुम्ही आधीच पेमेंट केले असेल तर हा संदेश दुर्लक्षित करा."
        ),
        "bengali": (
            "নমস্কার {name},\n\n"
            "আপনার {amount} পেমেন্ট সম্পূর্ণ হয়নি।\n\n"
            "নিচের নিরাপদ লিঙ্ক দিয়ে পেমেন্ট সম্পূর্ণ করুন:\n"
            f"{PAYMENT_LINK_PLACEHOLDER}\n\n"
            "আগেই পেমেন্ট করে থাকলে এই মেসেজটি উপেক্ষা করুন।"
        ),
        "gujarati": (
            "નમસ્તે {name},\n\n"
            "તમારું {amount} પેમેન્ટ પૂર્ણ થયું નથી.\n\n"
            "કૃપા કરીને નીચેની સુરક્ષિત લિંકથી પેમેન્ટ પૂર્ણ કરો:\n"
            f"{PAYMENT_LINK_PLACEHOLDER}\n\n"
            "જો તમે પહેલેથી પેમેન્ટ કર્યું હોય તો આ સંદેશ અવગણો."
        ),
        "kannada": (
            "ನಮಸ್ಕಾರ {name},\n\n"
            "ನಿಮ್ಮ {amount} ಪಾವತಿ ಪೂರ್ಣಗೊಂಡಿಲ್ಲ.\n\n"
            "ಕೆಳಗಿನ ಸುರಕ್ಷಿತ ಲಿಂಕ್ ಬಳಸಿ ಪಾವತಿ ಪೂರ್ಣಗೊಳಿಸಿ:\n"
            f"{PAYMENT_LINK_PLACEHOLDER}\n\n"
            "ನೀವು ಈಗಾಗಲೇ ಪಾವತಿ ಮಾಡಿದ್ದರೆ ಈ ಸಂದೇಶವನ್ನು ನಿರ್ಲಕ್ಷಿಸಿ."
        ),
        "malayalam": (
            "നമസ്കാരം {name},\n\n"
            "നിങ്ങളുടെ {amount} പേയ്‌മെന്റ് പൂർത്തിയായില്ല.\n\n"
            "താഴെയുള്ള സുരക്ഷിത ലിങ്ക് ഉപയോഗിച്ച് പേയ്‌മെന്റ് പൂർത്തിയാക്കുക:\n"
            f"{PAYMENT_LINK_PLACEHOLDER}\n\n"
            "നിങ്ങൾ ഇതിനകം അടച്ചിട്ടുണ്ടെങ്കിൽ ഈ സന്ദേശം അവഗണിക്കുക."
        ),
        "punjabi": (
            "ਸਤ ਸ੍ਰੀ ਅਕਾਲ {name},\n\n"
            "ਤੁਹਾਡੀ {amount} ਅਦਾਇਗੀ ਪੂਰੀ ਨਹੀਂ ਹੋ ਸਕੀ।\n\n"
            "ਹੇਠਾਂ ਦਿੱਤੇ ਸੁਰੱਖਿਅਤ ਲਿੰਕ ਨਾਲ ਅਦਾਇਗੀ ਪੂਰੀ ਕਰੋ:\n"
            f"{PAYMENT_LINK_PLACEHOLDER}\n\n"
            "ਜੇ ਤੁਸੀਂ ਪਹਿਲਾਂ ਹੀ ਅਦਾਇਗੀ ਕਰ ਚੁੱਕੇ ਹੋ ਤਾਂ ਇਸ ਸੁਨੇਹੇ ਨੂੰ ਨਜ਼ਰਅੰਦਾਜ਼ ਕਰੋ।"
        ),
    },
    "retry_payment": {
        "english": "Hi {name}, please retry your {amount} payment when you are ready.",
        "hinglish": "Hey {name}, jab ready ho aapka {amount} payment dobara try kijiye.",
        "hindi": "नमस्ते {name}, तैयार होने पर अपना {amount} भुगतान फिर से आज़माएँ।",
        "tamil": "வணக்கம் {name}, தயாரானதும் உங்கள் {amount} கட்டணத்தை மீண்டும் முயலவும்.",
        "telugu": "నమస్కారం {name}, సిద్ధమైనప్పుడు మీ {amount} చెల్లింపును మళ్లీ ప్రయత్నించండి.",
        "marathi": "नमस्कार {name}, तयार झाल्यावर तुमचे {amount} पेमेंट पुन्हा प्रयत्न करा.",
        "bengali": "নমস্কার {name}, প্রস্তুত হলে আপনার {amount} পেমেন্ট আবার চেষ্টা করুন।",
        "gujarati": "નમસ્તે {name}, તૈયાર થાઓ ત્યારે તમારું {amount} પેમેન્ટ ફરી અજમાવો.",
        "kannada": "ನಮಸ್ಕಾರ {name}, ಸಿದ್ಧರಾದಾಗ ನಿಮ್ಮ {amount} ಪಾವತಿಯನ್ನು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
        "malayalam": "നമസ്കാരം {name}, തയ്യാറാകുമ്പോൾ നിങ്ങളുടെ {amount} പേയ്‌മെന്റ് വീണ്ടും ശ്രമിക്കുക.",
        "punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ {name}, ਤਿਆਰ ਹੋਣ ਤੇ ਆਪਣੀ {amount} ਅਦਾਇਗੀ ਮੁੜ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
    },
    "whatsapp_reminder": {
        "english": "Hi {name}, a quick reminder that your {amount} payment is still pending.",
        "hinglish": "Hey {name}, reminder: aapka {amount} payment abhi pending hai.",
        "hindi": "नमस्ते {name}, याद दिलाना चाहेंगे कि आपका {amount} भुगतान अभी लंबित है।",
        "tamil": "வணக்கம் {name}, உங்கள் {amount} கட்டணம் இன்னும் நிலுவையில் உள்ளது.",
        "telugu": "నమస్కారం {name}, మీ {amount} చెల్లింపు ఇంకా పెండింగ్‌లో ఉంది.",
        "marathi": "नमस्कार {name}, तुमचे {amount} पेमेंट अजूनही प्रलंबित आहे.",
        "bengali": "নমস্কার {name}, আপনার {amount} পেমেন্ট এখনও বাকি আছে।",
        "gujarati": "નમસ્તે {name}, તમારું {amount} પેમેન્ટ હજુ પેન્ડિંગ છે.",
        "kannada": "ನಮಸ್ಕಾರ {name}, ನಿಮ್ಮ {amount} ಪಾವತಿ ಇನ್ನೂ ಬಾಕಿ ಇದೆ.",
        "malayalam": "നമസ്കാരം {name}, നിങ്ങളുടെ {amount} പേയ്‌മെന്റ് ഇപ്പഴും ബാക്കിയുണ്ട്.",
        "punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ {name}, ਤੁਹਾਡੀ {amount} ਅਦਾਇਗੀ ਅਜੇ ਵੀ ਬਾਕੀ ਹੈ।",
    },
    "escalate_to_merchant": {
        "english": "Hi {name}, we are reviewing your {amount} payment with the merchant team.",
        "hinglish": "Hey {name}, aapka {amount} payment merchant team review kar rahi hai.",
        "hindi": "नमस्ते {name}, आपकी {amount} भुगतान की समीक्षा व्यापारी टीम कर रही है।",
        "tamil": "வணக்கம் {name}, உங்கள் {amount} கட்டணத்தை வணிகர் குழு ஆய்வு செய்கிறது.",
        "telugu": "నమస్కారం {name}, మీ {amount} చెల్లింపును వ్యాపారి బృందం సమీక్షిస్తోంది.",
        "marathi": "नमस्कार {name}, तुमच्या {amount} पेमेंटची व्यापारी टीम समीक्षा करत आहे.",
        "bengali": "নমস্কার {name}, আপনার {amount} পেমেন্ট মার্চেন্ট টিম পর্যালোচনা করছে।",
        "gujarati": "નમસ્તે {name}, તમારા {amount} પેમેન્ટની સમીક્ષા મર્ચન્ટ ટીમ કરી રહી છે.",
        "kannada": "ನಮಸ್ಕಾರ {name}, ನಿಮ್ಮ {amount} ಪಾವತಿಯನ್ನು ವ್ಯಾಪಾರಿ ತಂಡ ಪರಿಶೀಲಿಸುತ್ತಿದೆ.",
        "malayalam": "നമസ്കാരം {name}, നിങ്ങളുടെ {amount} പേയ്‌മെന്റ് മർച്ചന്റ് ടീം അവലോകനം ചെയ്യുന്നു.",
        "punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ {name}, ਤੁਹਾਡੀ {amount} ਅਦਾਇਗੀ ਦੀ ਸਮੀਖਿਆ ਵਪਾਰੀ ਟੀਮ ਕਰ ਰਹੀ ਹੈ।",
    },
    "promise_to_pay": {
        "english": (
            "Thanks {name}. We noted your promise to pay {amount}. "
            "We will follow up only within policy limits."
        ),
        "hinglish": (
            "Thanks {name}. Aapka promise note ho gaya for {amount}. "
            "Hum sirf policy limits ke andar follow-up karenge."
        ),
        "hindi": (
            "धन्यवाद {name}. आपका {amount} भुगतान का वादा दर्ज किया गया। "
            "हम केवल नीति सीमाओं के भीतर अनुवर्ती करेंगे।"
        ),
        "tamil": (
            "நன்றி {name}. உங்கள் {amount} செலுத்தும் வாக்குறுதி பதிவு செய்யப்பட்டது. "
            "கொள்கை வரம்புக்குள் மட்டுமே தொடர்பு கொள்வோம்."
        ),
        "telugu": (
            "ధన్యవాదాలు {name}. మీ {amount} చెల్లింపు వాగ్దానం నమోదైంది. "
            "మేము విధాన పరిమితుల్లోనే ఫాలో అప్ చేస్తాము."
        ),
        "marathi": (
            "धन्यवाद {name}. तुमचे {amount} पेमेंटचे वचन नोंदले. "
            "आम्ही फक्त धोरण मर्यादेत फॉलो-अप करू."
        ),
        "bengali": (
            "ধন্যবাদ {name}. আপনার {amount} পরিশোধের প্রতিশ্রুতি নথিভুক্ত হয়েছে। "
            "আমরা শুধুমাত্র নীতির সীমার মধ্যে ফলোআপ করব।"
        ),
        "gujarati": (
            "આભાર {name}. તમારું {amount} ચૂકવવાનું વચન નોંધાયું. "
            "અમે ફક્ત નીતિ મર્યાદામાં જ ફોલો-અપ કરીશું."
        ),
        "kannada": (
            "ಧನ್ಯವಾದಗಳು {name}. ನಿಮ್ಮ {amount} ಪಾವತಿ ವಾಗ್ದಾನವನ್ನು ದಾಖಲಿಸಲಾಗಿದೆ. "
            "ನಾವು ನೀತಿ ಮಿತಿಯಲ್ಲಿ ಮಾತ್ರ ಅನುಸರಣೆ ಮಾಡುತ್ತೇವೆ."
        ),
        "malayalam": (
            "നന്ദി {name}. നിങ്ങളുടെ {amount} അടയ്ക്കുമെന്ന വാഗ്ദാനം രേഖപ്പെടുത്തി. "
            "നയ പരിധിക്കുള്ളിൽ മാത്രമേ ഞങ്ങൾ പിന്തുടരൂ."
        ),
        "punjabi": (
            "ਧੰਨਵਾਦ {name}. ਤੁਹਾਡਾ {amount} ਅਦਾਇਗੀ ਦਾ ਵਾਅਦਾ ਦਰਜ ਹੋ ਗਿਆ। "
            "ਅਸੀਂ ਸਿਰਫ਼ ਨੀਤੀ ਹੱਦਾਂ ਅੰਦਰ ਹੀ ਫਾਲੋ-ਅੱਪ ਕਰਾਂਗੇ।"
        ),
    },
}


def normalize_language(value: str | None) -> str:
    raw = (value or "english").strip().lower().replace("_", "-")
    aliases = {
        "en": "english",
        "eng": "english",
        "hi": "hindi",
        "hin": "hindi",
        "en-hi": "hinglish",
        "hi-en": "hinglish",
        "ta": "tamil",
        "te": "telugu",
        "mr": "marathi",
        "bn": "bengali",
        "gu": "gujarati",
        "kn": "kannada",
        "ml": "malayalam",
        "pa": "punjabi",
        "punjabi": "punjabi",
    }
    selected = aliases.get(raw, raw)
    return selected if selected in SUPPORTED_LANGUAGES else "english"


def list_supported_languages() -> list[dict[str, str]]:
    return [
        {"id": code, "label": LANGUAGE_LABELS.get(code, code.title())}
        for code in SUPPORTED_LANGUAGES
    ]


def render_template(
    action: str,
    *,
    language: str = "english",
    customer_name: str | None = None,
    amount_minor: int = 0,
    payment_link: str | None = None,
) -> dict[str, Any]:
    family = TEMPLATES.get(action) or TEMPLATES["whatsapp_reminder"]
    selected = normalize_language(language)
    if selected not in family:
        selected = "english"
    amount = f"₹{amount_minor / 100:,.0f}" if amount_minor >= 1000 else f"₹{amount_minor:,.0f}"
    # Application inserts the link. Models must never invent one.
    link_value = payment_link or "[payment link issued after approval]"
    message = family[selected].format(
        name=customer_name or "there",
        amount=amount,
    ).replace(PAYMENT_LINK_PLACEHOLDER, link_value)
    return {
        "intent": "customer_message",
        "language": selected,
        "message": message,
        "confidence": 1.0,
        "source": "deterministic_template",
        "action_context": action,
        "payment_link_inserted": bool(payment_link),
        "executed": False,
    }
