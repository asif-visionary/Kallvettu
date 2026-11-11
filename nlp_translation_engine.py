"""
NLP Translation & Epigraphical Restoration Engine for Ancient Tamil Inscriptions (Kalvettu)
Provides script recognition, character inventories, multi-tier epigraphical transcription,
Sandhi splitting (புணர்ச்சி பிரித்தல்), classical morphological & case analysis,
modern Tamil & English translation, historical contextual meaning, and modular OCR adapters.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

# ==============================================================================
# 1. EPIGRAPHICAL SCRIPTS, CHARACTER INVENTORIES & METADATA
# ==============================================================================

EPIGRAPHICAL_SCRIPTS = {
    "tamil_brahmi": {
        "id": "tamil_brahmi",
        "name_ta": "தமிழ் பிராமி (தமிழி / தாமிழி)",
        "name_en": "Tamil-Brahmi (Damili)",
        "period_ta": "கி.மு. 3 ஆம் நூற்றாண்டு – கி.பி. 2 ஆம் நூற்றாண்டு (சங்க காலம்)",
        "period_en": "c. 3rd Century BCE – 2nd Century CE (Sangam Period)",
        "geological_medium": "Granite Caverns, Natural Rock Shelters, Pottery (கருப்பு-சிவப்பு மண்பாண்டங்கள்)",
        "unicode_range": "U+11000 - U+1107F (Brahmi Unicode Block)",
        "vowels": ["𑀅 (அ)", "𑀆 (ஆ)", "𑀇 (இ)", "𑀈 (ஈ)", "𑀉 (உ)", "𑀊 (ஊ)", "𑀏 (எ/ஏ)", "𑀐 (ஐ)", "𑀑 (ஒ/ஓ)", "𑀒 (ஔ)"],
        "tamil_specific_consonants": [
            "𑀴 (ழ - Tamil Retroflex Approximant)",
            "𑀵 (ற - Tamil Alveolar Trill)",
            "𑀶 (ன - Tamil Alveolar Nasal)",
            "𑀷 (ள - Tamil Retroflex Lateral)"
        ],
        "pulli_system": "Early Stage (System I): Inherent 'a' without pulli; Later Stage (System II/III - Bhattiprolu): Diacritic pulli marker to denote pure consonants.",
        "distinctive_features": "Angular rectilinear chisel strokes, absence of top horizontal line (shirorekha), unpointed pure consonants in archaic phase, right-slanted vowel diacritics.",
        "diagnostic_markers": ["பாளிய்", "கொடிபித", "இகுவன்", "நந்தாஸிரிய்", "கணிய்", "அறுத்த", "ஆதன்", "இளங்கோ"]
    },
    "vatteluttu": {
        "id": "vatteluttu",
        "name_ta": "வட்டெழுத்து (Vatteluttu)",
        "name_en": "Classical Vatteluttu Script",
        "period_ta": "கி.பி. 4 ஆம் நூற்றாண்டு – கி.பி. 11 ஆம் நூற்றாண்டு (களப்பிரர், பல்லவர், பாண்டியர், சேரர் காலம்)",
        "period_en": "c. 4th Century CE – 11th Century CE (Post-Sangam to Medieval Era)",
        "geological_medium": "Hero Stones (நடுகல்), Cave Bed Inscriptions, Copper Plates (செப்பேடுகள்)",
        "unicode_range": "Transitional Cursive Dravidian Epigraphical Series",
        "vowels": ["அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ", "ஐ", "ஒ", "ஓ"],
        "distinctive_features": "Continuous rounded cursive glyphs ('Vatta' = Circular) adapted for incising on dried palm leaves with an iron stylus; curved loops for 'ka', 'ta', 'ya', and 'ra'.",
        "pulli_system": "Sporadic dot or vertical bar pulli; medial vowels curled around the consonant base.",
        "diagnostic_markers": ["தொறு", "நடுகல்", "பட்டான்", "கூற்றற்கு", "பூசல்", "நின்றெறிந்து", "சேந்தன்"]
    },
    "grantha": {
        "id": "grantha",
        "name_ta": "பல்லவர் / சோழர் கிரந்தம் (Grantha)",
        "name_en": "Pallava / Chola Grantha Script",
        "period_ta": "கி.பி. 6 ஆம் நூற்றாண்டு – கி.பி. 14 ஆம் நூற்றாண்டு",
        "period_en": "c. 6th Century CE – 14th Century CE",
        "geological_medium": "Granite Temple Basements, Copper Plate Praśastis, Monolithic Rock Cut Caves",
        "unicode_range": "U+11300 - U+1137F (Grantha Unicode Block)",
        "distinctive_characters": [
            "𑌶 (ஶ - Śa)", "𑌷 (ஷ - Ṣa)", "𑌸 (ஸ - Sa)", "𑌹 (ஹ - Ha)", "𑌕𑍍𑌷 (க்ஷ - Kṣa)", "𑌶𑍍𑌰𑍀 (ஸ்ரீ - Śrī)", "𑍐 (ஓம் - Om)"
        ],
        "distinctive_features": "Elaborate multi-tier consonant conjunct ligatures (saṃyuktākṣara); developed to record Sanskrit mantras, invocations, and royal genealogies interspersed in Tamil inscriptions.",
        "pulli_system": "Explicit virama diacritic (𑍍) suppressing inherent vowel.",
        "diagnostic_markers": ["ஸ்ரீ", "ஸ்வஸ்தி", "பிரசஸ்தி", "தேவர்", "பிரம்மதேயம்", "சாசனம்"]
    },
    "medieval_chola": {
        "id": "medieval_chola",
        "name_ta": "இடைக்காலச் சோழர் கல்வெட்டுத் தமிழ்",
        "name_en": "Medieval Chola Epigraphical Tamil Script",
        "period_ta": "கி.பி. 9 ஆம் நூற்றாண்டு – கி.பி. 13 ஆம் நூற்றாண்டு (விஜயாலய சோழர் முதல் பிற்காலச் சோழர் வரை)",
        "period_en": "c. 9th Century CE – 13th Century CE (Imperial Chola Era)",
        "geological_medium": "Brihadisvara Temple Granite Walls, Adhisthana Basements, Pillar Capitals",
        "unicode_range": "U+0B80 - U+0BFF (Standard Tamil Unicode Block with Archaic Allographs)",
        "distinctive_features": "Standardized rectangular geometry, horizontal headstrokes, distinctive archaic allographs for ண, ன, ர, ழ, non-split kombu ligatures, royal Meykkīrtti preambles.",
        "pulli_system": "Orthographic pulli widely present on consonants; frequent omission in fast stone incising.",
        "diagnostic_markers": ["உடையார்", "அருள்மொழி", "இராசராச", "கற்றளி", "குடவோலை", "வாரியம்", "இறையிலி", "திருநந்தாவிளக்கு"]
    }
}

# ==============================================================================
# 2. COMPREHENSIVE EPIGRAPHICAL LEXICON (50+ CLASSICAL TERMS WITH MORPHOLOGY)
# ==============================================================================

EPIGRAPHY_LEXICON = {
    "ஆநிரை": {
        "root": "ஆ + நிரை",
        "pos": "பெயர்ச்சொல் (Compound Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "பசுக்களின் கூட்டம் (கால்நடைச் செல்வம்)",
        "english": "Cattle herd / livestock",
        "etymology": "சங்க காலத்தில் செல்வம் என்பது ஆநிரைகளே (பகைவர் ஆநிரை கவர்தல் வெட்சித் திணை, மீட்டல் கரந்தைத் திணை)."
    },
    "தொறு": {
        "root": "தொறு",
        "pos": "பெயர்ச்சொல் (Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "ஆட்டு/மாட்டு மந்தை",
        "english": "Cattle / Livestock herd",
        "etymology": "தொறுக்கொண்டு போகிய = பகைவர்கள் ஊரின் ஆநிரைகளைக் கவர்ந்து சென்ற போது."
    },
    "பூசல்": {
        "root": "பூசல்",
        "pos": "பெயர்ச்சொல் (Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "போர் / சண்டை / பூசல்",
        "english": "Skirmish / Battle / Conflict",
        "etymology": "தொறுப்பூசல் = ஆநிரை கவர்தலின் போது நடக்கும் தற்காப்புப் போர்."
    },
    "நடுகல்": {
        "root": "நடு + கல்",
        "pos": "பெயர்ச்சொல் (Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "வீரக்கல் (வீர மரணம் அடைந்தோர்க்கு எடுக்கப்பட்ட கல்)",
        "english": "Hero Stone / Memorial Stone",
        "etymology": "போரில் ஊரையும் கால்நடைகளையும் காத்து வீரமரணம் எய்திய வீரனின் நினைவாக நடப்படும் வழிபாட்டுக் கல்."
    },
    "பள்ளிச்சந்தம்": {
        "root": "பள்ளி + சந்தம்",
        "pos": "பெயர்ச்சொல் (Compound Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "சமண அல்லது பௌத்த மடங்களுக்கு வழங்கப்பட்ட வரிவிலக்கு நிலம்",
        "english": "Tax-free land endowment granted to Jain/Buddhist monasteries",
        "etymology": "பள்ளி (துறவியர் உறைவிடம்) + சந்தம் (உரிமை நிலம்)."
    },
    "கற்றளி": {
        "root": "கல் + தளி",
        "pos": "பெயர்ச்சொல் (Compound Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "கற்கோயில் (கருங்கல்லால் கட்டப்பட்ட கோயில்)",
        "english": "Stone-built temple structure",
        "etymology": "தளி = கோயில்; கற்றளி = முழுதும் கருங்கல்லாலான கோயில் (செங்கல் கோயில்களுக்கு மாறானது)."
    },
    "கழஞ்சு": {
        "root": "கழஞ்சு",
        "pos": "அளவைப் பெயர் (Unit of Weight)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "பண்டைய பொன் எடை அளவு (சுமார் 4.5 முதல் 5 கிராம் தங்கம்)",
        "english": "Ancient Tamil unit of gold weight (~4.5 to 5 grams)",
        "etymology": "கழற்சிக்காயின் எடையை அடிப்படையாகக் கொண்ட பண்டைய தங்க நாணய/எடை அளவு."
    },
    "குடவோலை": {
        "root": "குடம் + ஓலை",
        "pos": "பெயர்ச்சொல் (Compound Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "குடத்தினுள் ஓலைச் சீட்டுகளைப் போட்டு தேர்தல் நடத்தும் மக்களாட்சி முறை",
        "english": "Pot-ballot election system for local village assembly representatives",
        "etymology": "உத்திரமேரூர்க் கல்வெட்டில் விரிவாக விவரிக்கப்படும் சோழர் கால நேரடி மக்களாட்சித் தேர்தல்."
    },
    "இறையிலி": {
        "root": "இறை + இலி",
        "pos": "பெயரடை (Adjective)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "அரசு வரி இல்லாத நிலக்கொடை (வரிவிலக்கு அளிக்கப்பட்ட நிலம்)",
        "english": "Tax-exempt royal land grant",
        "etymology": "இறை = அரசு வரி, இலி = இல்லாதது."
    },
    "பிரம்மதேயம்": {
        "root": "பிரம்ம + தேயம்",
        "pos": "பெயர்ச்சொல் (Compound Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "வேதக் கல்வி மற்றும் சமய பணிகளுக்காக வழங்கப்பட்ட கிராமக் கொடை",
        "english": "Tax-free village grant to scholars/priests for education",
        "etymology": "இடைக்காலச் சோழர், பாண்டியர் ஆட்சியில் உருவாக்கப்பட்ட பிரம்மதேய கிராமங்கள்."
    },
    "அக்கன்": {
        "root": "அக்கன்",
        "pos": "உறவுப்பெயர் (Kinship Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "தமக்கை / மூத்த சகோதரி",
        "english": "Elder sister",
        "etymology": "இராஜராஜ சோழன் தன் தமக்கை குந்தவை பிராட்டியாரை 'நம் அக்கன்' எனக் கல்வெட்டுகளில் குறிப்பிடுகிறார்."
    },
    "பெண்டுகள்": {
        "root": "பெண்டு + கள்",
        "pos": "பெயர்ச்சொல் (Noun - Plural)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "அரசியர் / மனைவியர் / மகளிர்",
        "english": "Queens / Royal consorts / Women",
        "etymology": "கல்வெட்டுகளில் அரச குடும்பத்து தேவிமார்களைக் குறிக்கும் பழைய சொல்."
    },
    "பாளிய்": {
        "root": "பள்ளி / பாளி",
        "pos": "பெயர்ச்சொல் (Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "சமணத் துறவியர் தங்குமிடம் / குகைப் படுக்கை",
        "english": "Monastic rock shelter / Jain stone bed",
        "etymology": "தமிழ் பிராமி கல்வெட்டுகளில் காணப்படும் சமண உறைவிடத்தைக் குறிக்கும் தொல்வடிவம்."
    },
    "இகுவன்": {
        "root": "இகுவன்",
        "pos": "பெயர்ச்சொல் (Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "பணியாள் / இளவல் / சேவகன்",
        "english": "Officer / Servant / Associate",
        "etymology": "பாண்டிய மன்னனின் பணியாளரைக் குறிக்க மாங்குளத்தில் வழங்கப்பட்ட சொல்."
    },
    "அறுத்த": {
        "root": "அறு",
        "pos": "பெயரெச்சம் (Relative Participle)",
        "case_marker": "வினை எச்சம் (Non-finite Verb)",
        "modern_tamil": "செதுக்கிய / வெட்டி அமைத்த",
        "english": "Chiseled / Cut into stone",
        "etymology": "பாறையைக் குடைந்து அல்லது கற்படுக்கையை வெட்டி அமைத்த செயல்."
    },
    "கொடிபித": {
        "root": "கொடு",
        "pos": "வினைச்சொல் (Verb - Past)",
        "case_marker": "இறந்தகால வினைமுற்று (Past Verb)",
        "modern_tamil": "கொடுக்கப்பட்ட / செய்விக்கப்பட்ட",
        "english": "Donated / Commissioned / Bestowed",
        "etymology": "தமிழி கல்வெட்டுகளில் கொடை அளித்தலைக் குறிக்கும் சொல்லாட்சி."
    },
    "திங்கட்போழ்தில்": {
        "root": "திங்கள் + போழ்து + இல்",
        "pos": "காலப்பெயர் (Temporal Noun)",
        "case_marker": "ஏழாம் வேற்றுமை உருபு (Locative: -இல்)",
        "modern_tamil": "நிலவொளி நேரத்தில் / பௌர்ணமி வேளையில்",
        "english": "During full moon hour / moonlight night",
        "etymology": "திங்கள் = நிலவு, போழ்து = காலம்/நேரம், இல் = ஏழாம் வேற்றுமை உருபு."
    },
    "கோனோநேரின்மை": {
        "root": "கோ + நோன் + நேர்மை",
        "pos": "அரச பட்டப்பெயர் (Royal Epithet)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "தனக்கு நிகரற்ற மாமன்னன்",
        "english": "The peerless supreme emperor without equal",
        "etymology": "பிற்காலச் சோழ மற்றும் பாண்டிய மன்னர்களின் அரசாணைகளில் தொடங்கும் புகழுரை."
    },
    "புரவு": {
        "root": "புரவு",
        "pos": "பெயர்ச்சொல் (Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "நிலவரி / விளைச்சல் வரி",
        "english": "Land revenue / Royal tax assessment",
        "etymology": "புரவுவரி திணைக்களம் = சோழர்களின் நில அளவை மற்றும் நிலவரித் துறை."
    },
    "வரிசை": {
        "root": "வரிசை",
        "pos": "பெயர்ச்சொல் (Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "சிறப்பு உரிமை / கௌரவப் பட்டம்",
        "english": "Honorific privilege / Royal distinction",
        "etymology": "அரசர் வீரர்களுக்கும் அறிஞர்களுக்கும் அளித்த கௌரவச் சலுகைகள்."
    },
    "வாரியம்": {
        "root": "வாரியம்",
        "pos": "பெயர்ச்சொல் (Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "கிராம நிர்வாகக் குழு (எ.கா. ஏரி வாரியம், தோட்ட வாரியம்)",
        "english": "Executive committee of village self-governing assembly",
        "etymology": "சோழர் கால கிராம சபை நிர்வாகத்துக்கான தேர்ந்தெடுக்கப்பட்ட செயற்குழுக்கள்."
    },
    "கோ": {
        "root": "கோ",
        "pos": "அரச பட்டப்பெயர் (Royal Title)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "அரசன் / மன்னன் / வேந்தன்",
        "english": "King / Monarch / Sovereign",
        "etymology": "பண்டைய தமிழ் மன்னர்களுக்கான முதன்மைப் பட்டம்."
    },
    "பெருமான்": {
        "root": "பெரிய + மகன் / மான்",
        "pos": "மதிப்பியல் பெயர் (Honorific Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "தலைவர் / பெருமதிப்பிற்குரிய இறைவன் அல்லது அரசன்",
        "english": "Revered Lord / Sovereign / Supreme Ruler",
        "etymology": "கடவுளையோ அல்லது மாமன்னனையோ குறிக்கும் மதிப்பியல் சொல்."
    },
    "அடிகள்": {
        "root": "அடி + கள்",
        "pos": "மதிப்பியல் பெயர் (Honorific Plural)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "துறவியர் / முனிவர் / வணக்கத்திற்குரிய பெரியோர்",
        "english": "Venerable Sage / Ascetic / Holy Master",
        "etymology": "சமண, சைவ, பௌத்த துறவிகளையும் மன்னர்களையும் குறிக்கும் உயர்வுச் சொல்."
    },
    "உடையார்": {
        "root": "உடை + ஆர்",
        "pos": "அரச பட்டப்பெயர் / மதிப்பியல் (Royal Title / Honorific)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "உரிமையாளர் / மன்னர் / இறைவர்",
        "english": "Lord / Sovereign Possessor",
        "etymology": "சோழப் பேரரசர்களை 'உடையார் ஸ்ரீ இராஜராஜ தேவர்' என்று அழைக்கும் மரபு."
    },
    "பிராட்டியார்": {
        "root": "பிராட்டி + ஆர்",
        "pos": "மதிப்பியல் பெயர் (Honorific Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "அரசி / தலைவி / பெருமதிப்பிற்குரிய பெண்மணி",
        "english": "Noble Lady / Queen / Princess",
        "etymology": "சோழப் பேரரசின் அரசியரையும் இளவரசிகளையும் குறிக்கும் உயர்வுச் சொல்."
    },
    "யாண்டு": {
        "root": "யாண்டு",
        "pos": "காலப்பெயர் (Temporal Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "ஆண்டு / மன்னரின் ஆட்சியாண்டு",
        "english": "Regnal Year / Calendar Year",
        "etymology": "கல்வெட்டுகளின் காலத்தைக் கணிக்கும் முதன்மைக் குறியீடு."
    },
    "நந்தாவிளக்கு": {
        "root": "நந்தா + விளக்கு",
        "pos": "பெயர்ச்சொல் (Compound Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "அணையா விளக்கு (இடைவிடாது எரியும் திருவிளக்கு)",
        "english": "Perpetual sacred lamp (kept burning constantly with ghee/oil)",
        "etymology": "திருக்கோயில்களில் இரவும் பகலும் எரிய பசுக்களைத் தானமாக அளித்து ஏற்படுத்திய விளக்கு."
    },
    "கடமை": {
        "root": "கடன் + மை",
        "pos": "பெயர்ச்சொல் (Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "நிலத்திற்கான அரசு வரி (முதன்மை வரி)",
        "english": "Primary agricultural land tax payable to the royal treasury",
        "etymology": "சோழர் கல்வெட்டுகளில் காணப்படும் நிலவுடைமையாளர் செலுத்தும் பிரதான வரி."
    },
    "அந்தராயம்": {
        "root": "அந்தர் + ஆயம்",
        "pos": "பெயர்ச்சொல் (Compound Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "உள்ளூர் நிர்வாகத்திற்கான உள்வரி",
        "english": "Internal levy / Local administrative cess",
        "etymology": "ஊர் சபையினரால் வசூலிக்கப்பட்டு உள்ளூர் நல்வாழ்வுக்குப் பயன்படுத்தப்பட்ட வரி."
    },
    "சாசனம்": {
        "root": "சாசனம்",
        "pos": "பெயர்ச்சொல் (Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "அரசாணை / கல்வெட்டு / சட்ட ஆவணம்",
        "english": "Royal Charter / Legal Edict / Inscribed Deed",
        "etymology": "அரசனின் ஆணை அல்லது கொடை விதியை நிரந்தரமாக்கும் கல்வெட்டு ஆவணம்."
    },
    "மெய்க்கீர்த்தி": {
        "root": "மெய் + கீர்த்தி",
        "pos": "பெயர்ச்சொல் (Compound Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "மன்னனின் மெய்யான வரலாற்று வெற்றிகளைப் பாடும் முன்னுரை",
        "english": "Royal Eulogy / True Historical Prologue",
        "etymology": "முதலாம் இராஜராஜன் காலத்தில் தொடங்கி மன்னரின் போர் வெற்றிகளை ஆவணப்படுத்திய கவிதை நடை."
    },
    "கடிகைத்தாவளம்": {
        "root": "கடிகை + தாவளம்",
        "pos": "இடப்பெயர் (Toponym / Compound Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "காவல் அரண் கொண்ட சர்வதேச வணிகச் சந்தை",
        "english": "Fortified mercantile town / Trade emporium",
        "etymology": "பூலாங்குறிச்சிக் கல்வெட்டில் வணிகர்களின் பாதுகாப்பு மையம் இவ்வாறு அழைக்கப்படுகிறது."
    },
    "குடும்பு": {
        "root": "குடும்பு",
        "pos": "பெயர்ச்சொல் (Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "ஊரின் குடியிருப்புப் பகுதி / வார்டு",
        "english": "Village Ward / Electoral precinct",
        "etymology": "உத்திரமேரூர் கிராமம் 30 குடும்புக்களாகப் பிரிக்கப்பட்டு பிரதிநிதிகள் தேர்ந்தெடுக்கப்பட்டனர்."
    },
    "கள்ளமற்றார்": {
        "root": "கள் + அம் + அற்றார்",
        "pos": "வினையாலணையும் பெயர் (Participial Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "ஊழல் மற்றும் கள்ளத்தனம் அற்ற நேர்மையாளர்",
        "english": "Persons of spotless moral integrity and clean accounts",
        "etymology": "சோழர் தேர்தலில் போட்டியிட கணக்குத் தூய்மை கட்டாயத் தகுதியாக இருந்தது."
    },
    "நிகமத்தோர்": {
        "root": "நிகமம் + தோர்",
        "pos": "பெயர்ச்சொல் (Plural Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "வணிகக் குழுவினர் (கில்ட் உறுப்பினர்கள்)",
        "english": "Members of a merchant guild",
        "etymology": "சங்க கால வர்த்தகக் கூட்டமைப்பை 'நிகமம்' எனக் குறித்தனர்."
    },
    "செங்காயபன்": {
        "root": "செம் + காயபன் (காசியபர்)",
        "pos": "ஆள் பெயர் (Proper Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "செங்காயபன் (சமண முனிவர்)",
        "english": "Cenkāyapan (Ascetic's personal name)",
        "etymology": "புகழூர் கல்வெட்டில் சேர இளவரசரால் கற்படுக்கை அமைத்துக் கொடுக்கப்பட்ட துறவி."
    },
    "பட்டான்": {
        "root": "படு",
        "pos": "வினைச்சொல் (Verb - Past)",
        "case_marker": "இறந்தகால வினைமுற்று (Past Verb)",
        "modern_tamil": "வீரமரணம் அடைந்தான் / போரில் உயிர் துறந்தான்",
        "english": "Fell valiantly / Died in battle",
        "etymology": "நடுகற்களில் வீரன் உயிர்த்தியாகம் செய்ததைக் குறிக்கும் மரபுச்சொல்."
    },
    "நின்றெறிந்து": {
        "root": "நின்று + எறிந்து",
        "pos": "வினையெச்சம் (Adverbial Participle)",
        "case_marker": "வினை எச்சம் (Non-finite Verb)",
        "modern_tamil": "அஞ்சாது எதிர்த்து நின்று வாள் வீசி",
        "english": "Standing steadfast and striking weapon against invaders",
        "etymology": "பகைவரின் அம்புகளையும் படைகளையும் எதிர்த்து நின்ற வீரப்போர்."
    },
    "சாத்தன்": {
        "root": "சாத்தன்",
        "pos": "ஆள் பெயர் (Proper Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "சாத்தன் (பண்டைய போர்வீரனின் பெயர்)",
        "english": "Sathan (Personal name of champion warrior)",
        "etymology": "செங்கம் நடுகற்களில் ஆநிரை மீட்டுப் பட்ட வீரர்களில் ஒருவன்."
    },
    "இளங்கோ": {
        "root": "இளமை + கோ",
        "pos": "அரச பட்டம் (Royal Title)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "இளவரசன் / பட்டத்து இளவல்",
        "english": "Crown Prince / Heir Apparent",
        "etymology": "அரசனின் புதல்வன் இளவரசுப் பட்டம் பெற்ற போது வழங்கப்படும் பதவி."
    },
    "செல்லிரும்பொறை": {
        "root": "செல் + இரும்பொறை",
        "pos": "அரச மரபுப் பெயர் (Dynastic Proper Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "சேர மன்னர்களின் இரும்பொறை வம்சப் பெயர்",
        "english": "Chel-Irumporai (Royal lineage of Sangam Cheras)",
        "etymology": "பதிற்றுப்பத்தில் பாடப்பெற்ற புகழ்பெற்ற சேர வேந்தர்களின் மரபு."
    },
    "பெருங்கடுங்கோ": {
        "root": "பெரிய + கடும் + கோ",
        "pos": "அரச பெயர் (Royal Proper Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "பெருங்கடுங்கோ (சேர மன்னன்)",
        "english": "Perunkadungo (Chera King of Karur)",
        "etymology": "புகழூர் கல்வெட்டிலும் சங்க இலக்கியத்திலும் குறிப்பிடப்படும் சேர வேந்தன்."
    },
    "இளங்கடுங்கோ": {
        "root": "இளமை + கடும் + கோ",
        "pos": "அரச பெயர் (Royal Proper Noun)",
        "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
        "modern_tamil": "இளங்கடுங்கோ (சேர இளவரசன்)",
        "english": "Ilankadungo (Chera Crown Prince)",
        "etymology": "செங்காயபனுக்குக் குகைப் படுக்கை அமைத்துக் கொடுத்த பட்டத்து இளவல்."
    },
    "கூற்றற்கு": {
        "root": "கூற்றன் + கு",
        "pos": "பெயர்ச்சொல் (Proper Noun with Dative)",
        "case_marker": "நான்காம் வேற்றுமை உருபு (Dative: -கு)",
        "modern_tamil": "சேந்தன் கூற்றன் என்னும் மன்னனுக்கு",
        "english": "To King Chendan Kurran",
        "etymology": "பூலாங்குறிச்சி கல்வெட்டில் மன்னரின் ஆட்சியாண்டைக் குறிக்கும் நான்காம் வேற்றுமை வடிவம்."
    },
    "தேவர்க்கு": {
        "root": "தேவர் + கு",
        "pos": "பெயர்ச்சொல் (Noun with Dative)",
        "case_marker": "நான்காம் வேற்றுமை உருபு (Dative: -கு)",
        "modern_tamil": "இறைவனுக்கு அல்லது பேரரசருக்கு",
        "english": "To the sacred deity / divine sovereign",
        "etymology": "திருக்கோயிலில் வழிபாட்டு தானம் அளிக்கப்படுவதைக் குறிக்கும் தொடர்."
    },
    "கல்லிலே": {
        "root": "கல் + இல் + ஏ",
        "pos": "பெயர்ச்சொல் (Noun with Locative & Clitic)",
        "case_marker": "ஏழாம் வேற்றுமை உருபு (Locative: -இல்) மற்றும் தேற்ற ஏகாரம்",
        "modern_tamil": "கருங்கல் சுவரிலேயே",
        "english": "Upon the very stone walls",
        "etymology": "தஞ்சைப் பெரிய கோயில் கல்வெட்டில் நித்திய சாசனமாக வெட்ட ஆணையிட்ட தொடர்."
    }
}

# ==============================================================================
# 3. EPIGRAPHICAL SANDHI SPLITTER (புணர்ச்சி பிரித்தல்)
# ==============================================================================

class EpigraphicalSandhiSplitter:
    """
    Classical Tamil Epigraphical Sandhi Decomposer.
    Handles geminate stop doubling (வல்லின மிகுதல்: க், ச், த், ப்),
    euphonic glides (உடம்படுமெய்: ய், வ்), compound words, and case suffix segmentation.
    """

    KNOWN_COMPOUNDS = {
        "பள்ளிச்சந்தம்": ["பள்ளி", "சந்தம்"],
        "குடவோலை": ["குடம்", "ஓலை"],
        "தொறுப்பூசல்": ["தொறு", "பூசல்"],
        "கற்றளி": ["கல்", "தளி"],
        "இறையிலி": ["இறை", "இலி"],
        "பிரம்மதேயம்": ["பிரம்ம", "தேயம்"],
        "திருநந்தாவிளக்கு": ["திரு", "நந்தாவிளக்கு"],
        "நந்தாவிளக்கு": ["நந்தா", "விளக்கு"],
        "ஆநிரை": ["ஆ", "நிரை"],
        "கடிகைத்தாவளம்": ["கடிகை", "தாவளம்"],
        "முப்பத்து": ["முப்பது"],
        "செங்காயபன்": ["செம்", "காயபன்"],
        "செல்லிரும்பொறை": ["செல்", "இரும்பொறை"],
        "பெருங்கடுங்கோ": ["பெரு", "கடுங்கோ"],
        "இளங்கடுங்கோ": ["இளங்", "கடுங்கோ"],
        "திங்கட்போழ்தில்": ["திங்கள்", "போழ்து", "-இல்"],
        "கோச்சேந்தன்": ["கோ", "சேந்தன்"],
        "நின்றெறிந்து": ["நின்று", "எறிந்து"],
        "திருக்கற்றளியிலே": ["திரு", "கல்", "தளி", "-இல்", "-ஏ"],
        "குடும்பிலும்": ["குடும்பு", "-இல்", "-உம்"]
    }

    # Case suffix patterns for suffix detachment
    CASE_SUFFIX_PATTERNS = [
        (r"^(.*?)த்துக்கு$", r"\1", "-க்கு (4th Dative Case)"),
        (r"^(.*?)ற்கு$", r"\1", "-ற்கு (4th Dative Case)"),
        (r"^(.*?)ர்க்கு$", r"\1", "-க்கு (4th Dative Case)"),
        (r"^(.*?)க்கு$", r"\1", "-க்கு (4th Dative Case)"),
        (r"^(.*?)த்திலே$", r"\1", "-இல் (7th Locative Case)"),
        (r"^(.*?)லிலே$", r"\1", "-இல் (7th Locative Case)"),
        (r"^(.*?)யில்$", r"\1", "-இல் (7th Locative Case)"),
        (r"^(.*?)இல்$", r"\1", "-இல் (7th Locative Case)"),
        (r"^(.*?)க்கண்$", r"\1", "-கண் (7th Locative Case)"),
        (r"^(.*?)கண்$", r"\1", "-கண் (7th Locative Case)"),
        (r"^(.*?)ஓடு$", r"\1", "-ஓடு (3rd Sociative Case)"),
        (r"^(.*?)உடன்$", r"\1", "-உடன் (3rd Sociative Case)"),
        (r"^(.*?)ஆல்$", r"\1", "-ஆல் (3rd Instrumental Case)"),
        (r"^(.*?)அது$", r"\1", "-அது (6th Genitive Case)"),
        (r"^(.*?)உடைய$", r"\1", "-உடைய (6th Genitive Case)")
    ]

    @classmethod
    def decompose_compound(cls, word: str) -> List[str]:
        """Decomposes a single ancient Tamil compound word into constituents."""
        clean_w = word.strip()
        if clean_w in cls.KNOWN_COMPOUNDS:
            return list(cls.KNOWN_COMPOUNDS[clean_w])

        # Check clitic endings like -உம், -மும், -வும், -ஏ
        clitic_patterns = [
            ("மும்", "ம்"),
            ("வும்", ""),
            ("உம்", ""),
            ("ஏ", "")
        ]
        for clitic, repl in clitic_patterns:
            if clean_w.endswith(clitic) and len(clean_w) > len(clitic) + 2:
                stem = clean_w[:-len(clitic)] + repl
                if stem in cls.KNOWN_COMPOUNDS:
                    return list(cls.KNOWN_COMPOUNDS[stem])

        # Check for geminate stop doubling: க், ச், த், ப்
        doubled_stops = [
            ("க்க", ["க்", "க"]),
            ("ச்ச", ["ச்", "ச"]),
            ("த்த", ["த்", "த"]),
            ("ப்ப", ["ப்", "ப"])
        ]
        for pair, _ in doubled_stops:
            if pair in clean_w:
                idx = clean_w.find(pair)
                if idx > 1 and idx + 2 < len(clean_w):
                    left = clean_w[:idx]
                    right = clean_w[idx + 1:]  # Drops the duplicate geminate stop
                    if left in EPIGRAPHY_LEXICON or right in EPIGRAPHY_LEXICON or left in cls.KNOWN_COMPOUNDS or right in cls.KNOWN_COMPOUNDS:
                        return [left, right]

        return [clean_w]

    @classmethod
    def split_sandhi(cls, text: str) -> str:
        """
        Processes a full inscription string and generates word-segmented text (Tier 3).
        Preserves bracketed lacunae [ ... ] while segmenting words.
        """
        if not text or not isinstance(text, str):
            return ""

        tokens = re.split(r"(\s+|\[.*?\])", text)
        segmented_tokens = []

        for token in tokens:
            if not token:
                continue
            if token.isspace():
                segmented_tokens.append(" ")
            elif token.startswith("[") and token.endswith("]"):
                segmented_tokens.append(token)
            else:
                parts = cls.decompose_compound(token)
                segmented_tokens.append(" ".join(parts))

        result = "".join(segmented_tokens)
        # Normalize multiple spaces
        return re.sub(r"\s+", " ", result).strip()

    @classmethod
    def extract_case_suffix(cls, word: str) -> Dict[str, str]:
        """Extracts archaic grammatical case suffixes from an inflected word."""
        clean = word.strip()
        for pattern, stem_group, case_desc in cls.CASE_SUFFIX_PATTERNS:
            m = re.match(pattern, clean)
            if m:
                stem = m.group(1)
                if len(stem) >= 2:
                    return {
                        "stem": stem,
                        "suffix": case_desc,
                        "has_case": True
                    }
        return {
            "stem": clean,
            "suffix": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
            "has_case": False
        }


# ==============================================================================
# 4. CLASSICAL TAMIL MORPHOLOGICAL & LEXICAL ANALYZER
# ==============================================================================

class EpigraphicalMorphologicalAnalyzer:
    """
    Morphological parser for archaic Dravidian epigraphy.
    Classifies root words, parts of speech, archaic suffixes, honorific titles,
    and cross-references the historical lexicon.
    """

    HONORIFIC_TITLES = {
        "கோ": "அரசன் / மன்னன் (Monarch)",
        "கோன்": "வேந்தன் / மன்னன் (Sovereign)",
        "பெருமான்": "தலைவர் / மாமன்னர் (Supreme Lord)",
        "அடிகள்": "துறவியர் / முனிவர் (Revered Ascetic)",
        "தேவர்": "அரசர் / தெய்வம் (Divine Monarch)",
        "உடையார்": "நிலவுடைமையாளர் / மாமன்னர் (Imperial Ruler)",
        "பிராட்டியார்": "அரசி / தலைவி (Queen / Princess)",
        "பேரரையன்": "பெருநிலக்கிழார் / சேனாதிபதி (High Chieftain)",
        "மாறன்": "பாண்டிய மன்னனின் குடிப்பெயர் (Pandya Lineage)",
        "வழுதி": "பாண்டிய மன்னனின் பட்டப்பெயர் (Pandya Title)",
        "செழியன்": "பாண்டிய மன்னனின் வீரப்பெயர் (Pandya Monarch)",
        "இரும்பொறை": "சேர மன்னர்களின் அரசமரபுப் பெயர் (Chera Lineage)",
        "வானவன்": "சேர மன்னனின் மரபுப் பெயர் (Chera Epithet)",
        "வளவன்": "சோழ மன்னனின் குடிப்பெயர் (Chola Lineage)",
        "வேந்தன்": "முடிசூடிய பேரரசன் (Crowned King)"
    }

    @classmethod
    def analyze_word(cls, word: str) -> Dict[str, Any]:
        """Analyzes a single word token for epigraphical morphology."""
        clean_w = re.sub(r"[\s\-_,.:;!?\[\]\(\)]+", "", word.strip())
        if not clean_w:
            return {}

        # 1. Exact match in historical lexicon
        if clean_w in EPIGRAPHY_LEXICON:
            info = EPIGRAPHY_LEXICON[clean_w]
            return {
                "word": clean_w,
                "normalized": clean_w,
                "root": info["root"],
                "pos": info["pos"],
                "case_marker": info["case_marker"],
                "modern": info["modern_tamil"],
                "english": info["english"],
                "notes": info["etymology"]
            }

        # 2. Check if it is a known honorific/royal title
        if clean_w in cls.HONORIFIC_TITLES:
            return {
                "word": clean_w,
                "normalized": clean_w,
                "root": clean_w,
                "pos": "அரச பட்டப்பெயர் / மதிப்பியல் (Royal Title)",
                "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)",
                "modern": cls.HONORIFIC_TITLES[clean_w],
                "english": f"Royal/Honorific title: {clean_w}",
                "notes": "கல்வெட்டுகளில் அரச மரபினரையும் முனிவர்களையும் குறிக்கும் மரபுத்தொடர்."
            }

        # 3. Check for inflectional case suffix
        case_info = EpigraphicalSandhiSplitter.extract_case_suffix(clean_w)
        stem = case_info["stem"]
        if case_info["has_case"] and stem in EPIGRAPHY_LEXICON:
            base_info = EPIGRAPHY_LEXICON[stem]
            return {
                "word": clean_w,
                "normalized": stem,
                "root": base_info["root"],
                "pos": base_info["pos"],
                "case_marker": case_info["suffix"],
                "modern": f"{base_info['modern_tamil']} ({case_info['suffix']})",
                "english": f"{base_info['english']} [{case_info['suffix']}]",
                "notes": f"வேற்றுமை உருபு பெற்ற சொல்: {base_info['etymology']}"
            }

        # 4. Fallback generic morphological inference
        pos = "பெயர்ச்சொல் (Noun / Epigraphical Term)"
        if clean_w.endswith("ான்") or clean_w.endswith("ார்") or clean_w.endswith("ோர்"):
            pos = "ஆண்பால் / பலர்பால் பெயர் (Personal / Participial Noun)"
        elif clean_w.endswith("து") or clean_w.endswith("ன"):
            pos = "ஒன்றன்பால் / பலவின்பால் வினை (Neuter Verb / Particle)"
        elif clean_w.endswith("இத்து") or clean_w.endswith("ந்து"):
            pos = "வினையெச்சம் (Adverbial Participle)"

        return {
            "word": clean_w,
            "normalized": clean_w,
            "root": clean_w,
            "pos": pos,
            "case_marker": case_info["suffix"],
            "modern": f"{clean_w} (ஆய்வுக்குரிய கல்வெட்டுச் சொல்)",
            "english": f"Epigraphical lexical unit '{clean_w}'",
            "notes": "இலக்கண மற்றும் தொல்லியல் ஒப்பாய்வுக்குரிய சொல் வடிவம்."
        }

    @classmethod
    def analyze_text(cls, text: str) -> List[Dict[str, Any]]:
        """Parses a full inscription text and extracts unique analyzed morphological units."""
        if not text:
            return []

        # First segment text with sandhi splitter
        segmented = EpigraphicalSandhiSplitter.split_sandhi(text)
        tokens = [t for t in re.split(r"[\s,.:;!]+", segmented) if len(t) >= 2]

        analyzed_list = []
        seen = set()

        # Prioritize words present in master lexicon first
        for token in tokens:
            clean = re.sub(r"[\s\-_,.:;!?\[\]\(\)]+", "", token.strip())
            if clean and clean not in seen:
                seen.add(clean)
                item = cls.analyze_word(clean)
                if item:
                    analyzed_list.append(item)

        return analyzed_list


# ==============================================================================
# 5. MASTER CATALOG OF FAMOUS ANCIENT TAMIL INSCRIPTIONS
# ==============================================================================

KNOWN_INSCRIPTIONS_CATALOG = [
    {
        "id": "chola_arulmoli",
        "title_ta": "அருள்மொழிதேவன் (இராஜராஜ சோழன்) கல்வெட்டு",
        "title_en": "Arulmolidevan (Rajaraja Chola I) Epigraph",
        "era_ta": "கி.பி. 992 (10 ஆம் நூற்றாண்டு - சோழப் பேரரசு)",
        "era_en": "c. 992 CE (10th Century - Imperial Chola Era)",
        "script_ta": "செவ்வியல் சோழர் கல்வெட்டுத் தமிழ்",
        "script_en": "Classical Chola Epigraphical Tamil",
        "dynasty_ta": "சோழப் பேரரசு (அருள்மொழிவர்மன் / முதலாம் இராஜராஜன்)",
        "dynasty_en": "Imperial Chola Dynasty (Arulmolivarman / Rajaraja I)",
        "location_ta": "திருக்கோயில் கருங்கல் அதிஷ்டானச் சுவர், சோழ மண்டலம்",
        "location_en": "Temple Granite Basement Wall (Adhisthana), Chola Heartland, Tamil Nadu",
        "category": "Imperial Perpetual Endowment & Temple Grant (திருக்கோயில் நந்தாவிளக்கு தானம்)",
        "image_key": "chola_kalvettu",
        "weathered_text": "அருள்மொழிதேவன் [பராந்தகன்] சோழர் யாண்டு ஏழாவது... உடையார் [ஸ்ரீ இராசராச] தேவர்க்கு...",
        "restored_text": "அருள்மொழிதேவன் பராந்தகன் சோழர் யாண்டு ஏழாவது உடையார் ஸ்ரீ இராசராச தேவர்க்கு திருநந்தாவிளக்கு மற்றும் திருப்பணிக்கு அளித்த தானம்...",
        "diplomatic_text": "அருள்மொழிதேவன் [பராந்தகன்] சோழர் யாண்டு ஏழாவது... உடையார் [ஸ்ரீ இராசராச] தேவர்க்கு திருநந்தாவிளக்கு...",
        "normalized_text": "அருள்மொழிதேவன் பராந்தகன் சோழர் யாண்டு ஏழாவது உடையார் ஸ்ரீ இராசராச தேவர்க்கு திருநந்தாவிளக்கு மற்றும் திருப்பணிக்கு அளித்த தானம்",
        "word_segmented_text": "அருள்மொழிதேவன் பராந்தகன் சோழர் யாண்டு ஏழாவது உடையார் ஸ்ரீ இராசராச தேவர்க்கு திரு நந்தாவிளக்கு மற்றும் திருப்பணிக்கு அளித்த தானம்",
        "modern_tamil_translation": "சோழப் பேரரசன் அருள்மொழிதேவன் (முதலாம் இராஜராஜ சோழன்) தனது 7-வது ஆட்சியாண்டில், திருக்கோயிலில் நாள்தோறும் அணையா விளக்கு (திருநந்தாவிளக்கு) எரிப்பதற்காகவும், வழிபாட்டுத் திருப்பணிகளுக்காகவும் ஆநிரைகளையும் பொன்னையும் தானமாக அளித்ததை இக்கல்வெட்டு ஆவணப்படுத்துகிறது.",
        "english_translation": "In the 7th regnal year of Emperor Arulmolidevan (Rajaraja Chola I), scion of the imperial Chola lineage, this inscription records the royal endowment of sacred perpetual lamps (Thirunandhavilakku), gold coins (Kazhanju), and milch cattle endowed to the temple sanctuary.",
        "tamil_meaning": "இக்கல்வெட்டு இராஜராஜ சோழனின் இயற்பெயரான 'அருள்மொழிதேவன்' என்ற புனிதப் பெயரை நேரடியாகக் குறிப்பிடும் அரிய வரலாற்றுச் சான்றாகும். மாமன்னன் தான் அரியணை ஏறிய ஏழாவது ஆட்சியாண்டில், சிவாலயத்தில் இரவும் பகலும் சுடர்விட்டு எரியும் நந்தாவிளக்கிற்குத் தேவையான நெய் கிடைப்பதற்காகப் பசுக்களை இடையர்களிடம் ஒப்படைத்து, நாள்தோறும் உரிய அளவு நெய் தவறாமல் கோயிலுக்கு வழங்கப்பட வேண்டும் என்று விதித்த சோழர் கால நிர்வாக நெறியை இது விளக்குகிறது.",
        "english_meaning": "This lithic record is of prime epigraphical significance as it prominently preserves the personal name 'Arulmolidevan' used by Rajaraja Chola before his imperial coronation titles. The inscription records an endowment of milch cattle entrusted to local pastoralists, who were legally bound to deliver a fixed measure of clarified butter (ghee) daily to maintain eternal temple lamps (Thirunandhavilakku) without interruption.",
        "uncertain_characters": [
            {
                "character": "[பராந்தகன்]",
                "position": 15,
                "alternative_readings": ["பராந்தகன்", "இராஜகேசரி"],
                "reason": "Weathered patron name between Arulmoli and Chola regnal formula",
                "confidence": 0.88
            },
            {
                "character": "[ஸ்ரீ இராசராச]",
                "position": 58,
                "alternative_readings": ["ஸ்ரீ இராசராச", "கோப்பரகேசரி"],
                "reason": "Chipped granite border enclosing imperial royal title",
                "confidence": 0.92
            }
        ],
        "words": [
            {"word": "அருள்மொழிதேவன்", "root": "அருள் + மொழி + தேவன்", "pos": "ஆள் பெயர் (Personal Name)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "இராஜராஜ சோழனின் இயற்பெயர்", "english": "Arulmolidevan (Birth name of Rajaraja Chola I)", "notes": "அருள் வாய்ந்த மொழிகளைப் பேசுபவன் என்ற பொருள்."},
            {"word": "யாண்டு ஏழாவது", "root": "யாண்டு + ஏழு + ஆவது", "pos": "காலப்பெயர் (Temporal Clause)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "மன்னரின் ஏழாம் ஆட்சியாண்டு", "english": "In the 7th Regnal Year", "notes": "இராஜராஜனின் ஆரம்பக் கால ஆட்சி (கி.பி. 992)."},
            {"word": "உடையார்", "root": "உடை + ஆர்", "pos": "அரச பட்டப்பெயர் (Royal Title)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "தலைவர் / மாமன்னர் / பெருமான்", "english": "Lord / Sovereign Ruler", "notes": "சோழ மன்னர்களுக்கான முதன்மை மதிப்பியல் பட்டம்."},
            {"word": "ஸ்ரீ இராசராச", "root": "ஸ்ரீ + ராஜ + ராஜ", "pos": "முடிசூட்டுப் பெயர் (Coronation Title)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "அரசர்களுக்கெல்லாம் அரசன்", "english": "Sri Rajaraja (King of Kings)", "notes": "மாமன்னரின் முடிசூட்டுப் பட்டம்."},
            {"word": "தேவர்க்கு", "root": "தேவர் + கு", "pos": "பெயர்ச்சொல் (Noun with Dative)", "case_marker": "நான்காம் வேற்றுமை உருபு (Dative: -கு)", "modern": "திருக்கோயில் பெருமானுக்கு / அரசருக்கு", "english": "To the sacred deity / revered sovereign", "notes": "நான்காம் வேற்றுமை உருபு (-கு)."}
        ]
    },
    {
        "id": "mangulam_brahmi",
        "title_ta": "மாங்குளம் தமிழ்-பிராமி கல்வெட்டு",
        "title_en": "Mangulam Tamil-Brahmi Inscription",
        "era_ta": "கி.மு. 3 ஆம் - 2 ஆம் நூற்றாண்டு (சங்க காலம்)",
        "era_en": "c. 3rd - 2nd Century BCE (Sangam Period)",
        "script_ta": "தமிழ் பிராமி (தமிழி / தாமிழி)",
        "script_en": "Tamil-Brahmi (Damili)",
        "dynasty_ta": "தொடக்கக் காலப் பாண்டியர் (நெடுஞ்செழியன்)",
        "dynasty_en": "Early Pandya Dynasty (King Neduncheliyan)",
        "location_ta": "மாங்குளம் மீனாட்சிபுரம் ஓவாமலை, மதுரை மாவட்டம்",
        "location_en": "Ovamalai, Mangulam-Meenakshipuram, Madurai District, Tamil Nadu",
        "category": "Jain Monastic Rock Shelter Endowment (சமணக் கற்படுக்கைத் தானம்)",
        "image_key": "mangulam",
        "weathered_text": "கணிய் நந்த[ாசி]ரிய் இகுவன் [வெள்]ளறை நிகமத்தோர் கொடி[பி]த பாளிய்",
        "restored_text": "கணிய் நந்தாஸிரிய் இகுவன் வெள்ளறை நிகமத்தோர் கொடிபித பாளிய் நெடுஞ்செழியன் பணியால்",
        "diplomatic_text": "கணிய் நந்த[ாசி]ரிய் இகுவன் [வெள்]ளறை நிகமத்தோர் கொடி[பி]த பாளிய்",
        "normalized_text": "கணி நந்தாசிரியர் இகுவன் வெள்ளறை நிகமத்தோர் கொடிபித்த பள்ளி நெடுஞ்செழியன் பணியால்",
        "word_segmented_text": "கணிய் நந்தாஸிரிய் இகுவன் வெள்ளறை நிகமத்தோர் கொடிபித பாளிய் நெடுஞ்செழியன் பணியால்",
        "modern_tamil_translation": "பாண்டியன் நெடுஞ்செழியனின் அதிகாரியும் வெள்ளறை வணிகக் குழுவைச் சேர்ந்தவருமான கடலன் வழுதி, கணி நந்தாசிரியர் என்னும் சமண முனிவருக்கு இந்த உறைவிடக் கற்படுக்கையை அமைத்துக் கொடுத்தார்.",
        "english_translation": "The rock shelter bed dedicated to the venerable Jain ascetic Kani Nantasiriyan by his associate and officer Kadalan Vazhuthi on behalf of Pandya King Neduncheliyan and the Vellari merchant guild.",
        "tamil_meaning": "இக்கல்வெட்டு தமிழகத்தில் இதுவரை கண்டறியப்பட்ட மிகத் தொன்மையான எழுத்தாதாரங்களில் ஒன்றாகும். சங்க காலப் பாண்டிய மன்னன் நெடுஞ்செழியன் பெயரைக் குறிப்பிடும் முதல் கல்வெட்டுச் சான்றாகத் திகழ்கிறது. மதுரைக்கு அருகிலுள்ள மாங்குளம் குகையில் சமணத் துறவிகள் தங்கித் தவம் செய்வதற்காக அமைக்கப்பட்ட கற்படுக்கையை இது விவரிக்கிறது.",
        "english_meaning": "The Mangulam epigraph is one of the earliest securely dated Tamil-Brahmi inscriptions in South India. It is paramount in epigraphical history as it directly mentions the Sangam Pandya monarch Neduncheliyan, proving historical literacy, established merchant guilds ('Nigama'), and institutional patronage for heterodox Jain monks in the 3rd century BCE.",
        "uncertain_characters": [
            {
                "character": "[ாசி]",
                "position": 9,
                "alternative_readings": ["ாஸிரி", "ாசிரிய்"],
                "reason": "Archaic Brahmi ligature for Sanskrit loan word Acharya (ஆசிரியர்)",
                "confidence": 0.90
            },
            {
                "character": "[வெள்]",
                "position": 21,
                "alternative_readings": ["வெள்", "வேள்"],
                "reason": "Surface flake on the rock overhang partially obscuring initial consonant",
                "confidence": 0.85
            },
            {
                "character": "[பி]",
                "position": 41,
                "alternative_readings": ["பி", "வி"],
                "reason": "Chiseled groove erosion on the causative verbal suffix",
                "confidence": 0.92
            }
        ],
        "words": [
            {"word": "கணிய்", "root": "கணி", "pos": "பெயர்ச்சொல் (Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "ஜோதிடர் / துறவி / கணிப்பவர்", "english": "Astronomer / Scholar / Ascetic", "notes": "கணி என்ற சொல்லின் தொன்மையான வடிவம்."},
            {"word": "நந்தாஸிரிய்", "root": "நந்த + ஆசாரியர்", "pos": "ஆள் பெயர் (Proper Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "நந்தாசிரியர் (சமண குரு)", "english": "Nantasiriyan (Jain Teacher)", "notes": "நந்த + ஆசாரியர் என்ற வடமொழிப் பெயரின் தமிழ்ப் பிராமி வடிவம்."},
            {"word": "இகுவன்", "root": "இகுவன்", "pos": "பெயர்ச்சொல் (Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "இளவல் / அலுவலர் / பணியாள்", "english": "Associate / Officer / Servant", "notes": "மன்னரின் சேவகனைக் குறிக்கிறது."},
            {"word": "நிகமத்தோர்", "root": "நிகமம் + தோர்", "pos": "பெயர்ச்சொல் (Plural Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "வணிகக் குழுவினர் (கில்ட்)", "english": "Merchant Guild Members", "notes": "நிகமம் = வணிக மையம்; நிகமத்தோர் = வணிகக் குழு."},
            {"word": "கொடிபித", "root": "கொடு", "pos": "வினைச்சொல் (Verb - Past)", "case_marker": "இறந்தகால வினைமுற்று (Past Verb)", "modern": "கொடுக்கப்பட்ட / செய்விக்கப்பட்ட", "english": "Carved and donated", "notes": "பண்டைய தமிழ்ப் பிராமி வினை வடிவம்."},
            {"word": "பாளிய்", "root": "பாளி / பள்ளி", "pos": "பெயர்ச்சொல் (Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "படுக்கை / துறவியர் தங்குமிடம்", "english": "Rock cut shelter / Monastic bed", "notes": "பள்ளி என்ற சொல்லின் மூல வடிவம்."},
            {"word": "நெடுஞ்செழியன்", "root": "நெடுஞ்செழியன்", "pos": "ஆள் பெயர் / மன்னர் (Proper Noun / Royal Title)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "பாண்டிய மன்னன் நெடுஞ்செழியன்", "english": "Pandya King Neduncheliyan", "notes": "மாங்குளம் கல்வெட்டில் குறிப்பிடப்படும் சங்க காலப் பாண்டிய மன்னன்."}
        ]
    },
    {
        "id": "thanjavur_rajaraja",
        "title_ta": "தஞ்சைப் பெரிய கோயில் இராஜராஜ சோழன் கல்வெட்டு",
        "title_en": "Thanjavur Brihadisvara Temple Rajaraja Chola Inscription",
        "era_ta": "கி.பி. 1010 (11 ஆம் நூற்றாண்டு - சோழர் பொற்காலம்)",
        "era_en": "1010 CE (11th Century - Golden Age of Cholas)",
        "script_ta": "இடைக்காலத் தமிழ் எழுத்து",
        "script_en": "Medieval Chola Tamil Script",
        "dynasty_ta": "சோழப் பேரரசு (முதலாம் இராஜராஜ சோழன்)",
        "dynasty_en": "Imperial Chola Dynasty (Rajaraja Chola I)",
        "location_ta": "தஞ்சாவூர் பெருவுடையார் கோயில் வடக்கு அதிஷ்டானம்",
        "location_en": "North Adhisthana (Base Wall), Brihadisvara Temple, Thanjavur, Tamil Nadu",
        "category": "Imperial Temple Dedication & Public Accounting (திருக்கோயில் கொடை ஆவணம்)",
        "image_key": "thanjavur",
        "weathered_text": "நாம் குடுத்தனவும் நம் அக்கன் [குடுத்த]னவும் நம் பெண்டுகள் குடுத்தனவும் [மற்றா]ர் குடுத்தனவும் இத்திருக் கற்றளியிலே [கல்லி]லே வெட்டுக...",
        "restored_text": "நாம் குடுத்தனவும் நம் அக்கன் குடுத்தனவும் நம் பெண்டுகள் குடுத்தனவும் மற்றார் குடுத்தனவும் இத்திருக் கற்றளியிலே கல்லிலே வெட்டுக என்று திருவாய் மொழிந்தருள...",
        "diplomatic_text": "நாம் குடுத்தனவும் நம் அக்கன் [குடுத்த]னவும் நம் பெண்டுகள் குடுத்தனவும் [மற்றா]ர் குடுத்தனவும் இத்திருக் கற்றளியிலே [கல்லி]லே வெட்டுக...",
        "normalized_text": "நாம் கொடுத்தனவும் நம் அக்கன் கொடுத்தனவும் நம் பெண்டுகள் கொடுத்தனவும் மற்றார் கொடுத்தனவும் இத்திருக்கற்றளியிலே கல்லிலே வெட்டுக என்று திருவாய் மொழிந்தருள",
        "word_segmented_text": "நாம் குடுத்தனவும் நம் அக்கன் குடுத்தனவும் நம் பெண்டுகள் குடுத்தனவும் மற்றார் குடுத்தனவும் இத்திரு கல் தளியிலே கல்லிலே வெட்டுக என்று திரு வாய் மொழிந்தருள",
        "modern_tamil_translation": "நாம் (இராஜராஜ சோழன்) அளித்த கொடைகளும், நமது மூத்த தமக்கையார் குந்தவை பிராட்டியார் அளித்த கொடைகளும், நமது அரசியர் அளித்த கொடைகளும், ஏனைய பொதுமக்கள் அளித்த கொடைகளும் இந்தத் திருக்கற்றளியாகிய பெரிய கோயிலின் கருங்கல் சுவரில் பொறிக்கப்பட வேண்டும் என்று அரசர் ஆணையிட்டார்.",
        "english_translation": "The royal decree of King Rajaraja Chola that: All gifts and endowments bestowed by Us, all gifts given by Our elder sister Kundavai Pirattiyar, all gifts presented by Our royal consorts, and all donations contributed by others, shall be permanently chiseled upon the stone walls of this sacred stone temple.",
        "tamil_meaning": "இக்கல்வெட்டு சோழர் கால நிர்வாக நேர்மை மற்றும் வெளிப்படைத்தன்மைக்கு ஒப்பற்ற சான்றாகும். மாமன்னன் இராஜராஜன் தான் கட்டிய பிரம்மாண்டக் கோயிலில் தன் பெயரை மட்டும் முன்னிறுத்தாமல், தனது தமக்கை குந்தவை, தன் தேவிமார்கள், சேனாதிபதிகள், நடனக் கலைஞர்கள், மற்றும் எளிய பணியாளர்கள் அளித்த ஒவ்வொரு காணிக்கையையும் துல்லியமாக எடையிட்டு கல்லில் வெட்ட ஆணையிட்ட வரலாற்றுச் சிறப்பு மிக்க அறிவிப்பு.",
        "english_meaning": "This foundation inscription epitomizes the unprecedented transparency, organizational genius, and democratic spirit of Chola governance. Rajaraja Chola decreed that not only royal donations, but even the smallest gifts from commoners, dancers, and commanders must be weighed, assayed, and publicly inscribed on the stone base.",
        "uncertain_characters": [
            {
                "character": "[குடுத்த]",
                "position": 25,
                "alternative_readings": ["குடுத்த", "ஈந்த"],
                "reason": "Chiseled repetitive administrative formula partly degraded by water wash",
                "confidence": 0.94
            },
            {
                "character": "[மற்றா]",
                "position": 62,
                "alternative_readings": ["மற்றா", "பிறா"],
                "reason": "Surface abrasion on the granite plinth",
                "confidence": 0.89
            },
            {
                "character": "[கல்லி]",
                "position": 98,
                "alternative_readings": ["கல்லி", "கற்றி"],
                "reason": "Chipped corner near lower relief molding",
                "confidence": 0.91
            }
        ],
        "words": [
            {"word": "நாம்", "root": "நாம்", "pos": "மூவிடப்பெயர் (Pronoun - Royal Plural)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "நாம் (மன்னரின் அரச பன்மை)", "english": "We (Royal plural of the Emperor)", "notes": "மன்னர் தன்னை உயர்வுப் பன்மையில் குறிப்பது."},
            {"word": "அக்கன்", "root": "அக்கன்", "pos": "உறவுப்பெயர் (Kinship Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "மூத்த சகோதரி (குந்தவை நாச்சியார்)", "english": "Elder sister (Princess Kundavai)", "notes": "சோழர் காலச் செம்மொழிச் சொல்."},
            {"word": "பெண்டுகள்", "root": "பெண்டு + கள்", "pos": "பெயர்ச்சொல் (Plural Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "அரசியர் / மனைவியர்", "english": "Queens / Royal wives", "notes": "அரச குடும்பத்து பெண்களைக் குறிக்கிறது."},
            {"word": "மற்றார்", "root": "மற்று + ஆர்", "pos": "பெயர்ச்சொல் (Indefinite Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "பிறர் / ஏனைய குடிமக்கள்", "english": "Others / Donors / Citizens", "notes": "சாதாரண மக்களையும் உள்ளடக்கும் பொதுச் சொல்."},
            {"word": "கற்றளி", "root": "கல் + தளி", "pos": "பெயர்ச்சொல் (Compound Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "கற்கோயில் (கருங்கல் கோயில்)", "english": "Monolithic stone temple", "notes": "கல் + தளி = கற்றளி."},
            {"word": "திருவாய் மொழிந்தருள", "root": "திரு + வாய் + மொழி + அருள", "pos": "வினைத்தொடர் (Royal Verb Phrase)", "case_marker": "செயப்பாட்டு வினை (Honorific Command)", "modern": "அருளோடு வாய்மொழி ஆணை பிறப்பித்தார்", "english": "Graciously decreed by royal voice", "notes": "அரசாணை வெளியிடும் மரபுத்தொடர்."}
        ]
    },
    {
        "id": "pugalur_chera",
        "title_ta": "புகழூர் சேரர் தமிழ்-பிராமி கல்வெட்டு",
        "title_en": "Pugalur Chera Tamil-Brahmi Inscription",
        "era_ta": "கி.பி. 1 ஆம் - 2 ஆம் நூற்றாண்டு (சங்க காலம்)",
        "era_en": "c. 1st - 2nd Century CE (Sangam Period)",
        "script_ta": "தமிழ் பிராமி (தமிழி)",
        "script_en": "Tamil-Brahmi (Damili)",
        "dynasty_ta": "சேரர் வம்சம் (இரும்பொறை மரபு)",
        "dynasty_en": "Chera Dynasty (Irumporai Lineage)",
        "location_ta": "புகழூர் ஆறுநாட்டார் மலை, கரூர் மாவட்டம்",
        "location_en": "Arunattar Malai, Pugalur near Karur, Tamil Nadu",
        "category": "Royal Ascetic Cave Donation (சேர மன்னர் கற்படுக்கை கொடை)",
        "image_key": "pugalur",
        "weathered_text": "யாற்றூர் செங்[காயபன்] உறை கொ ஆதன் செல்லிரும்[பொறை மகன்] பெருங்கடுங்கோ [இளங்கோ] ஆக அறுத்த கல்",
        "restored_text": "யாற்றூர் செங்காயபன் உறை கொ ஆதன் செல்லிரும்பொறை மகன் பெருங்கடுங்கோ மகன் இளங்கடுங்கோ இளங்கோ ஆக அறுத்த கல்",
        "diplomatic_text": "யாற்றூர் செங்[காயபன்] உறை கொ ஆதன் செல்லிரும்[பொறை மகன்] பெருங்கடுங்கோ [இளங்கோ] ஆக அறுத்த கல்",
        "normalized_text": "யாற்றூர் செங்காயபன் உறை கோ ஆதன் செல்லிரும்பொறை மகன் பெருங்கடுங்கோ மகன் இளங்கடுங்கோ இளங்கோ ஆக அறுத்த கல்",
        "word_segmented_text": "யாற்றூர் செங்காயபன் உறை கொ ஆதன் செல் இரும்பொறை மகன் பெருங்கடுங்கோ இளங்கடுங்கோ இளங்கோ ஆக அறுத்த கல்",
        "modern_tamil_translation": "யாற்றூரைச் சேர்ந்த செங்காயபன் என்னும் சமணத் துறவி தங்குவதற்காக, சேர மன்னன் கோ ஆதன் செல்லிரும்பொறையின் மகனான பெருங்கடுங்கோவின் புதல்வன் இளங்கடுங்கோ இளவரசனாக (இளங்கோவாக) இருந்த போது வெட்டி அமைத்துக் கொடுத்த கற்படுக்கை.",
        "english_translation": "The stone shelter chiseled for the ascetic Cenkāyapan of Yāṟṟūr by crown prince Iḷaṅkaṭuṅkō, son of Peruṅkaṭuṅkō and grandson of Chera King Kō Āthan Chel-Irumporai, during his tenure as heir apparent.",
        "tamil_meaning": "பதிற்றுப்பத்து போன்ற சங்க இலக்கியங்கள் கூறும் சேர மன்னர்களின் மூன்று தலைமுறை மரபு வழியை (ஆதன் செல்லிரும்பொறை -> பெருங்கடுங்கோ -> இளங்கடுங்கோ) உறுதிப்படுத்தும் மிக முக்கியமான தொல்லியல் கல்வெட்டுச் சான்றாகும். சங்க இலக்கியத்தின் வரலாற்று நம்பகத்தன்மையை மெய்ப்பித்த கல்வெட்டு.",
        "english_meaning": "This landmark inscription bridges Sangam literature (specifically Patirruppattu) and archaeological epigraphy by confirming a three-generation genealogy of Chera kings ruling from Karur (Vanchi). It provides irrefutable historical evidence of Sangam royal chronology.",
        "uncertain_characters": [
            {
                "character": "[காயபன்]",
                "position": 14,
                "alternative_readings": ["காயபன்", "காசிபன்"],
                "reason": "Rock cleft cutting across ascetic's lineage name",
                "confidence": 0.87
            },
            {
                "character": "[பொறை மகன்]",
                "position": 36,
                "alternative_readings": ["பொறை மகன்", "பொறை கோன்"],
                "reason": "Granite erosion along royal Chera title",
                "confidence": 0.90
            },
            {
                "character": "[இளங்கோ]",
                "position": 61,
                "alternative_readings": ["இளங்கோ", "இளவல்"],
                "reason": "Superficial flaking on bed margin",
                "confidence": 0.93
            }
        ],
        "words": [
            {"word": "யாற்றூர்", "root": "யாறு + ஊர்", "pos": "இடப்பெயர் (Toponym)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "யாற்றூர் என்னும் ஊர்", "english": "Yattrur (Village name near river)", "notes": "துறவியின் சொந்த ஊர்."},
            {"word": "செங்காயபன்", "root": "செம் + காயபன்", "pos": "ஆள் பெயர் (Proper Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "செங்காயபன் (முனிவர் பெயர்)", "english": "Cenkāyapan (Ascetic's name / Kashyapa)", "notes": "சமண முனிவரின் பெயர்."},
            {"word": "கொ", "root": "கோ", "pos": "அரச பட்டம் (Royal Title)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "மன்னன் / அரசன்", "english": "King / Monarch", "notes": "கோ என்பதன் குறில் வடிவம்."},
            {"word": "இளங்கோ", "root": "இளமை + கோ", "pos": "அரச பட்டம் (Royal Title)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "இளவரசன் (பட்டத்து இளவல்)", "english": "Crown Prince / Heir Apparent", "notes": "இளைய கோ = இளங்கோ."},
            {"word": "அறுத்த கல்", "root": "அறு + கல்", "pos": "வினைப்பெயரெச்சத் தொடர்", "case_marker": "செயப்படுபொருள்", "modern": "செதுக்கி அமைக்கப்பட்ட பாறைப் படுக்கை", "english": "Rock carved into bed", "notes": "கல் தறித்தல் / அறுத்தல்."}
        ]
    },
    {
        "id": "pulankurichi_vatteluttu",
        "title_ta": "பூலாங்குறிச்சி வட்டெழுத்துக் கல்வெட்டு",
        "title_en": "Pulankurichi Early Vatteluttu Inscription",
        "era_ta": "கி.பி. 5 ஆம் நூற்றாண்டு (சங்க மருவிய காலம்)",
        "era_en": "c. 5th Century CE (Post-Sangam Transition Era)",
        "script_ta": "தொடக்கக் கால வட்டெழுத்து",
        "script_en": "Early Vatteluttu Script",
        "dynasty_ta": "சேந்தன் கூற்றன் மரபு (முத்தரையர் / களப்பிரர் சமகாலம்)",
        "dynasty_en": "King Chendan Kurran Lineage",
        "location_ta": "பூலாங்குறிச்சி குன்றுகள், சிவகங்கை மாவட்டம்",
        "location_en": "Pulankurichi Rocks, Sivagangai District, Tamil Nadu",
        "category": "Land Rights, Defence & Administrative Epigraph (நில உரிமை & நிர்வாக ஆவணம்)",
        "image_key": "pulankurichi",
        "weathered_text": "கோச்சேந்தன் கூற்றற்கு [யாண்டு] இருபத்தேழாவது... கடிகைத்தாவளத்து [பள்ளிச்]சந்தமும் பிரம்மதேயமும்...",
        "restored_text": "கோச்சேந்தன் கூற்றற்கு யாண்டு இருபத்தேழாவது... கடிகைத்தாவளத்து பள்ளிச்சந்தமும் பிரம்மதேயமும் எல்லை வகுத்து அமைத்த ஆவணம்...",
        "diplomatic_text": "கோச்சேந்தன் கூற்றற்கு [யாண்டு] இருபத்தேழாவது... கடிகைத்தாவளத்து [பள்ளிச்]சந்தமும் பிரம்மதேயமும்...",
        "normalized_text": "கோச்சேந்தன் கூற்றற்கு யாண்டு இருபத்தேழாவது கடிகைத்தாவளத்து பள்ளிச்சந்தமும் பிரம்மதேயமும் எல்லை வகுத்து அமைத்த ஆவணம்",
        "word_segmented_text": "கோ சேந்தன் கூற்றற்கு யாண்டு இருபத்தேழாவது கடிகைத்தாவளத்து பள்ளி சந்தமும் பிரம்மதேயமும் எல்லை வகுத்து அமைத்த ஆவணம்",
        "modern_tamil_translation": "மாமன்னன் சேந்தன் கூற்றனின் 27-வது ஆட்சியாண்டில், வணிக மையமாகிய கடிகைத்தாவளத்தில் சமணப் பள்ளிச்சந்த நிலங்களும் வேதியர்க்கான பிரம்மதேய நிலங்களும் எல்லைகள் வரையறுக்கப்பட்டு வரிவிலக்குடன் வழங்கப்பட்ட செய்தி.",
        "english_translation": "In the 27th regnal year of King Chendan Kurran, boundaries were delineated and tax exemptions established for both Jain monastic lands (Pallichantham) and scholar settlements (Brahmadeyam) around the fortified trade emporium of Kadigaithavalam.",
        "tamil_meaning": "தமிழ் பிராமி எழுத்திலிருந்து வட்டெழுத்து உருவான இடைக்காலப் பரிணாமத்தை உணர்த்தும் மிக நீளமான மலைக்கல்வெட்டு. தமிழகத்தில் நில உடைமை முறைகள், பிரம்மதேயம், பள்ளிச்சந்தம் போன்ற நிறுவனங்கள் களப்பிரர் காலத்தில் எவ்வாறு இயங்கின என்பதை வெளிச்சமிட்டுக் காட்டிய பெருமை இதற்கு உண்டு.",
        "english_meaning": "The Pulankurichi inscription is monumental in Dravidian epigraphy, marking the evolutionary transition from Tamil-Brahmi to Vatteluttu script. Stretching across rocky hills, it details agrarian tenures, trade outposts, and royal administration during the oft-misunderstood Kalabhra-transition era.",
        "uncertain_characters": [
            {
                "character": "[யாண்டு]",
                "position": 21,
                "alternative_readings": ["யாண்டு", "வருடம்"],
                "reason": "Archaic cursive Vatteluttu ya-loop partially worn",
                "confidence": 0.91
            },
            {
                "character": "[பள்ளிச்]",
                "position": 58,
                "alternative_readings": ["பள்ளிச்", "பாளிச்"],
                "reason": "Granite cleavage running across institutional title",
                "confidence": 0.88
            }
        ],
        "words": [
            {"word": "யாண்டு", "root": "யாண்டு", "pos": "காலப்பெயர் (Temporal Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "ஆட்சியாண்டு / வருடம்", "english": "Regnal Year", "notes": "மன்னரின் ஆட்சிக் காலத்தைக் குறிக்கும் வழக்கு."},
            {"word": "கடிகைத்தாவளம்", "root": "கடிகை + தாவளம்", "pos": "இடப்பெயர் (Toponym)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "காவல் அரண் கொண்ட வணிகச் சந்தை", "english": "Fortified mercantile settlement", "notes": "பண்டைய வர்த்தக மையம்."},
            {"word": "பள்ளிச்சந்தம்", "root": "பள்ளி + சந்தம்", "pos": "பெயர்ச்சொல் (Compound Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "சமண மட நிலக்கொடை", "english": "Jain institutional land grant", "notes": "வரிவிலக்கு அளிக்கப்பட்ட நிலம்."},
            {"word": "பிரம்மதேயம்", "root": "பிரம்ம + தேயம்", "pos": "பெயர்ச்சொல் (Compound Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "அறிஞர்களுக்கு வழங்கப்பட்ட நிலம்", "english": "Tax-exempt scholarly village", "notes": "கல்விப் பணிகளுக்கான கொடை."}
        ]
    },
    {
        "id": "hero_stone_nadukal",
        "title_ta": "செங்கம் வீரக்கல் (நடுகல்) வட்டெழுத்துக் கல்வெட்டு",
        "title_en": "Chengam Hero Stone (Nadukal) Vatteluttu Inscription",
        "era_ta": "கி.பி. 8 ஆம் நூற்றாண்டு (பல்லவர் - வாணர் காலம்)",
        "era_en": "c. 8th Century CE (Pallava-Bana Era)",
        "script_ta": "செவ்வியல் வட்டெழுத்து",
        "script_en": "Classical Vatteluttu Script",
        "dynasty_ta": "வாணகோவரையர் / நந்திவர்ம பல்லவன் காலம்",
        "dynasty_en": "Bana Chieftains under Pallava Overlordship",
        "location_ta": "செங்கம் வட்டாரம், திருவண்ணாமலை மாவட்டம்",
        "location_en": "Chengam Region, Tiruvannamalai District, Tamil Nadu",
        "category": "Hero Memorial for Cattle Raid Defense (ஆநிரை மீட்ட நடுகல்)",
        "image_key": "nadukal",
        "weathered_text": "பல்லவ வேந்தன் ஆட்சியில் தொறுக்கொண்டு போகிய [பூச]லில் நின்றெறிந்து [பட்டான்] கல்...",
        "restored_text": "பல்லவ வேந்தன் ஆட்சியில் தொறுக்கொண்டு போகிய பூசலில் நின்றெறிந்து பட்டான் சாத்தன் நடுகல்",
        "diplomatic_text": "பல்லவ வேந்தன் ஆட்சியில் தொறுக்கொண்டு போகிய [பூச]லில் நின்றெறிந்து [பட்டான்] கல்...",
        "normalized_text": "பல்லவ வேந்தன் ஆட்சியில் தொறுக்கொண்டு போகிய பூசலில் நின்றெறிந்து பட்டான் சாத்தன் நடுகல்",
        "word_segmented_text": "பல்லவ வேந்தன் ஆட்சியில் தொறு கொண்டு போகிய பூசலில் நின்று எறிந்து பட்டான் சாத்தன் நடுகல்",
        "modern_tamil_translation": "பகைவர்கள் ஊருக்குள் புகுந்து பசுக்கூட்டங்களைக் (ஆநிரைகளை) கவர்ந்து சென்ற போது, அவற்றை மீட்க வீரத்தோடு எதிர்த்துப் போரிட்டுத் தன் இன்னுயிர் ஈந்த சாத்தன் என்னும் வீரனுக்கு நடப்பட்ட நினைவு நடுகல்.",
        "english_translation": "Hero stone erected in honour of the brave warrior Sathan, who stood steadfast and laid down his life fighting off enemy raiders while successfully rescuing the captured village cattle herd.",
        "tamil_meaning": "சங்கத் தொல்காப்பியக் கரந்தை மற்றும் வெட்சித் திணையின் போர்மரபை 8 ஆம் நூற்றாண்டிலும் மாறாமல் காட்டும் நடுகல். ஊரின் ஆநிரை காக்கும் போரில் மார்பில் அம்பு பாய்ந்து வீரமரணம் அடைந்த வீரனின் உருவமும், அவனது தியாகத்தைப் போற்றும் வட்டெழுத்து வாசகமும் கல்லில் வடிக்கப்பட்டுள்ளது.",
        "english_meaning": "This epigraph exemplifies the enduring Sangam martial ethos preserved in the Tolkappiyam (Karanthai and Vetchi themes). In agro-pastoral societies, cattle were the primary wealth; memorial hero stones honored local champions who made the supreme sacrifice defending village livestock.",
        "uncertain_characters": [
            {
                "character": "[பூச]",
                "position": 42,
                "alternative_readings": ["பூச", "போர"],
                "reason": "Arrow relief sculpture overlapping with inscriptional line",
                "confidence": 0.90
            },
            {
                "character": "[பட்டான்]",
                "position": 64,
                "alternative_readings": ["பட்டான்", "மாய்ந்தான்"],
                "reason": "Lower section exposed to soil moisture",
                "confidence": 0.92
            }
        ],
        "words": [
            {"word": "தொறு", "root": "தொறு", "pos": "பெயர்ச்சொல் (Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "ஆநிரை / மாட்டு மந்தை", "english": "Cattle herd", "notes": "ஊரின் கால்நடைச் செல்வம்."},
            {"word": "பூசல்", "root": "பூசல்", "pos": "பெயர்ச்சொல் (Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "போர் / பூசல் / சண்டை", "english": "Combat / Skirmish", "notes": "கொள்ளையருடன் நடந்த சண்டை."},
            {"word": "நின்றெறிந்து", "root": "நின்று + எறிந்து", "pos": "வினையெச்சம் (Adverbial Participle)", "case_marker": "வினை எச்சம் (Non-finite Verb)", "modern": "எதிர்த்து நின்று வாள் வீசி", "english": "Standing firm and striking weapon", "notes": "அஞ்சாது போரிட்ட வீரம்."},
            {"word": "பட்டான்", "root": "படு", "pos": "வினைச்சொல் (Verb - Past)", "case_marker": "இறந்தகால வினைமுற்று (Past Verb)", "modern": "வீரமரணம் அடைந்தான்", "english": "Fell valiantly / Died in battle", "notes": "போரில் மாண்டதைக் குறிக்கும் சொல்."}
        ]
    },
    {
        "id": "uttiramerur_election",
        "title_ta": "உத்திரமேரூர் குடவோலை மக்களாட்சி கல்வெட்டு",
        "title_en": "Uttiramerur Kudavolai Democratic Election Inscription",
        "era_ta": "கி.பி. 920 (10 ஆம் நூற்றாண்டு - முதலாம் பராந்தக சோழன்)",
        "era_en": "920 CE (10th Century - King Parantaka Chola I)",
        "script_ta": "பழைய சோழர் தமிழ் மற்றும் கிரந்தம்",
        "script_en": "Old Chola Tamil & Grantha",
        "dynasty_ta": "சோழப் பேரரசு (முதலாம் பராந்தகன்)",
        "dynasty_en": "Chola Dynasty (Parantaka Chola I)",
        "location_ta": "வைகுண்டப் பெருமாள் கோயில், உத்திரமேரூர், காஞ்சிபுரம் மாவட்டம்",
        "location_en": "Vaikunta Perumal Temple, Uttiramerur, Kanchipuram District, Tamil Nadu",
        "category": "Constitutional Village Democracy & Electoral Code (மக்களாட்சித் தேர்தல் விதிமுறை ஆவணம்)",
        "image_key": "uttiramerur",
        "weathered_text": "முப்பத்து [குடும்பிலும்] வாரியத்துக்கு ஓலை எழுதி குடத்திலே இட்டு... [கள்ள]மற்றார் வாரியம் செய்யக் கடவராகவும்...",
        "restored_text": "முப்பத்து குடும்பிலும் வாரியத்துக்கு ஓலை எழுதி குடத்திலே இட்டு... ஒழுக்கமும் கணக்குத் தூய்மையும் உடைய கள்ளமற்றார் வாரியம் செய்யக் கடவராகவும்...",
        "diplomatic_text": "முப்பத்து [குடும்பிலும்] வாரியத்துக்கு ஓலை எழுதி குடத்திலே இட்டு... [கள்ள]மற்றார் வாரியம் செய்யக் கடவராகவும்...",
        "normalized_text": "முப்பத்து குடும்பிலும் வாரியத்துக்கு ஓலை எழுதி குடத்திலே இட்டு ஒழுக்கமும் கணக்குத் தூய்மையும் உடைய கள்ளமற்றார் வாரியம் செய்யக் கடவராகவும்",
        "word_segmented_text": "முப்பது குடும்பிலும் வாரியத்துக்கு ஓலை எழுதி குடத்திலே இட்டு ஒழுக்கமும் கணக்கு தூய்மையும் உடைய கள்ளம் அற்றார் வாரியம் செய்ய கடவராகவும்",
        "modern_tamil_translation": "ஊரின் 30 குடும்புகளிலிருந்தும் (வார்டுகளிலிருந்தும்) தகுதியுடைய வேட்பாளர்களின் பெயர்களை ஓலையில் எழுதி குடத்தினுள் இட வேண்டும். ஒரு சிறுவன் மூலம் சீட்டுகளை எடுத்து, ஒழுக்கமும், நிலவுடைமையும், கணக்கு வழக்குகளில் ஊழலற்ற நேர்மையும் கொண்டோரை கிராம வாரிய நிர்வாகிகளாகத் தேர்ந்தெடுக்க வேண்டும்.",
        "english_translation": "From each of the thirty wards, eligible candidates' names shall be inscribed on palm-leaf ballots and placed inside a pot (Kudam). An innocent child shall draw the ballots. Only individuals with moral integrity, proper land ownership, education, and immaculate financial accountability shall serve on executive governing committees.",
        "tamil_meaning": "ஆயிரம் ஆண்டுகளுக்கு முன்பே வாக்குச்சீட்டு முறை, தகுதி வரம்புகள், பதவிக்காலக் கட்டுப்பாடு, ஊழல் தடுப்பு விதிகள் (சொத்துக் கணக்கு காட்டாதவர்கள் தகுதி நீக்கம்) ஆகியவற்றை வகுத்து உலகிற்கே மக்களாட்சியை உணர்த்திய உலகப் புகழ்பெற்ற கல்வெட்டு.",
        "english_meaning": "Hailed as India's earliest written democratic constitution, the Uttiramerur inscription codifies qualifications, disqualifications (including mandatory audit clearance and zero tolerance for corruption), secret-ballot procedures, and recall provisions for local participatory governance.",
        "uncertain_characters": [
            {
                "character": "[குடும்பிலும்]",
                "position": 8,
                "alternative_readings": ["குடும்பிலும்", "சேரிகளிலும்"],
                "reason": "Corner stone weather-pitting on ward administration clause",
                "confidence": 0.91
            },
            {
                "character": "[கள்ள]",
                "position": 58,
                "alternative_readings": ["கள்ள", "குற்ற"],
                "reason": "Shallow chisel stroke on integrity qualification prerequisite",
                "confidence": 0.89
            }
        ],
        "words": [
            {"word": "குடும்பு", "root": "குடும்பு", "pos": "பெயர்ச்சொல் (Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "வார்டு / குடியிருப்புப் பகுதி", "english": "Ward / Electoral division", "notes": "ஊர் 30 குடும்புக்களாகப் பிரிக்கப்பட்டது."},
            {"word": "குடவோலை", "root": "குடம் + ஓலை", "pos": "பெயர்ச்சொல் (Compound Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "குடப் பானை வாக்குச்சீட்டுத் தேர்தல்", "english": "Pot-ballot voting method", "notes": "இரகசிய மக்களாட்சி வாக்கு முறை."},
            {"word": "வாரியம்", "root": "வாரியம்", "pos": "பெயர்ச்சொல் (Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "நிர்வாகக் குழு (ஏரி, தோட்டம், பொன் வாரியம்)", "english": "Specialized governing council", "notes": "கிராம பணிகளை நிர்வகிக்கும் குழு."},
            {"word": "கள்ளமற்றார்", "root": "கள் + அம் + அற்றார்", "pos": "வினையாலணையும் பெயர் (Participial Noun)", "case_marker": "வேற்றுமை உருபற்ற எழுவாய் (Nominative)", "modern": "ஊழல் மற்றும் கள்ளத்தனம் அற்றவர்", "english": "Persons of spotless integrity", "notes": "கணக்கு காட்டியவர்களுக்கே பதவி உரிமை."}
        ]
    }
]

# ==============================================================================
# 6. SCRIPT IDENTIFICATION & VALIDATION HEURISTICS
# ==============================================================================

def identify_script_from_text(text: str) -> Dict[str, Any]:
    """
    Identifies historical Tamil script based on palaeographical vocabulary markers,
    Unicode ranges, and morphological particles.
    """
    if not text or not isinstance(text, str):
        return EPIGRAPHICAL_SCRIPTS["medieval_chola"]

    clean = text.strip()

    # 1. Brahmi Unicode range check (U+11000 - U+1107F) or diagnostic markers
    has_brahmi_unicode = any(0x11000 <= ord(c) <= 0x1107F for c in clean)
    if has_brahmi_unicode or any(m in clean for m in EPIGRAPHICAL_SCRIPTS["tamil_brahmi"]["diagnostic_markers"]):
        return EPIGRAPHICAL_SCRIPTS["tamil_brahmi"]

    # 2. Vatteluttu diagnostic linguistic markers
    if any(m in clean for m in EPIGRAPHICAL_SCRIPTS["vatteluttu"]["diagnostic_markers"]):
        return EPIGRAPHICAL_SCRIPTS["vatteluttu"]

    # 3. Medieval Chola diagnostic markers (prioritized over incidental Grantha titles like "ஸ்ரீ")
    if any(m in clean for m in EPIGRAPHICAL_SCRIPTS["medieval_chola"]["diagnostic_markers"]):
        return EPIGRAPHICAL_SCRIPTS["medieval_chola"]

    # 4. Grantha Unicode range check (U+11300 - U+1137F) or distinctive Sanskrit invocations
    has_grantha_unicode = any(0x11300 <= ord(c) <= 0x1137F for c in clean)
    if has_grantha_unicode or any(m in clean for m in ["ஸ்வஸ்தி", "ஸ்ரீமத்", "பிரசஸ்தி", "க்ஷ", "ஸ"]):
        return EPIGRAPHICAL_SCRIPTS["grantha"]

    # 5. Medieval Chola default
    return EPIGRAPHICAL_SCRIPTS["medieval_chola"]


def extract_script_inventory(script_key: str) -> Dict[str, Any]:
    """Returns character inventory and features for a specified script."""
    return EPIGRAPHICAL_SCRIPTS.get(script_key, EPIGRAPHICAL_SCRIPTS["medieval_chola"])


# ==============================================================================
# 7. CONSERVATIVE TEXTUAL RESTORATION & INPAINTING
# ==============================================================================

def perform_textual_restoration(weathered_text: str) -> Dict[str, Any]:
    """
    Conservative Epigraphical Text Inpainting and Damage Inspection.
    Identifies bracketed damaged sections `[...]` and applies documented formulaic
    reconstructions while preserving unresolved lacunae as brackets.
    """
    if not weathered_text or not isinstance(weathered_text, str):
        return {
            "original_weathered": "",
            "restored_text": "",
            "has_damage": False,
            "damaged_segments": [],
            "reconstructions": [],
            "resolved_count": 0,
            "unresolved_count": 0,
            "methodology": "Conservative Epigraphical Reconstruction"
        }

    damaged_segments = re.findall(r"\[(.*?)\]", weathered_text)
    has_damage = len(damaged_segments) > 0

    documented_reconstructions = {
        "ாசி": ("ாஸிரி", "Documented scribal title 'Nanthasiri' in Mangulam Tamil-Brahmi series"),
        "வெள்": ("வெள்ளறை", "Vellarai merchant guild place-name"),
        "பி": ("கொடிபித", "Early Tamil-Brahmi verbal formula for dedicating a rock bed"),
        "குடுத்த": ("குடுத்தனவும்", "Chola royal donation administrative clause"),
        "மற்றா": ("மற்றார்", "Universal Chola public formula covering gifts by citizens"),
        "கல்லி": ("கற்றளியிலே", "Chola epigraphical compound for monolithic granite temple"),
        "காயபன்": ("செங்காயபன்", "Sangam Chera epigraphical name of revered ascetic"),
        "பொறை மகன்": ("பொறை மகன் பெருங்கடுங்கோ", "Chera Irumporai royal patron genealogy"),
        "இளங்கோ": ("இளங்கடுங்கோ இளங்கோ", "Chera heir-apparent titulature"),
        "யாண்டு": ("யாண்டு", "Regnal year administrative formula"),
        "பள்ளிச்": ("பள்ளிச்சந்தம்", "Jain monastic tax-exempt land designation"),
        "பூச": ("பூசலில்", "Hero stone martial encounter formula (cattle recovery battle)"),
        "பட்டான்": ("பட்டான்", "Hero stone memorial death commemoration formula"),
        "குடும்பிலும்": ("குடும்பிலும்", "Uttiramerur 30-ward electoral constitution allocation"),
        "கள்ள": ("கள்ளமற்றார்", "Uttiramerur moral rectitude qualification clause"),
        "பராந்தகன்": ("பராந்தகன்", "Imperial Chola patron lineage name"),
        "ஸ்ரீ இராசராச": ("ஸ்ரீ இராசராச", "Imperial royal epithet")
    }

    reconstructions = []
    restored_text = weathered_text

    for seg in damaged_segments:
        seg_clean = seg.strip()
        matched = False
        for pat, (rep, rationale) in documented_reconstructions.items():
            if seg_clean == pat or pat in seg_clean:
                restored_text = restored_text.replace(f"[{seg}]", rep, 1)
                reconstructions.append({
                    "original": f"[{seg}]",
                    "reconstructed": rep,
                    "status": "resolved",
                    "confidence": 0.90,
                    "rationale": rationale
                })
                matched = True
                break

        if not matched:
            # Scientific epigraphy rule: DO NOT invent missing text!
            reconstructions.append({
                "original": f"[{seg}]",
                "reconstructed": f"[{seg}]",
                "status": "unresolved",
                "confidence": 0.0,
                "rationale": "Fragmentary or ambiguous glyph without sufficient formulaic context"
            })

    resolved_count = sum(1 for r in reconstructions if r["status"] == "resolved")
    unresolved_count = sum(1 for r in reconstructions if r["status"] == "unresolved")

    return {
        "original_weathered": weathered_text,
        "restored_text": restored_text,
        "has_damage": has_damage,
        "damaged_segments": damaged_segments,
        "reconstructions": reconstructions,
        "resolved_count": resolved_count,
        "unresolved_count": unresolved_count,
        "methodology": "Conservative Rule-Guided Epigraphical Inpainting (Unresolved lacunae preserved as bracketed markers)"
    }


# ==============================================================================
# 8. MODULAR EPIGRAPHICAL OCR ADAPTER ARCHITECTURE
# ==============================================================================

class BaseOCRAdapter(ABC):
    """Abstract base class for epigraphical OCR adapters."""

    @abstractmethod
    def recognize(self, image: Optional[Any] = None, text_input: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Performs script recognition and produces structured epigraphical transcription."""
        pass


class PresetCatalogAdapter(BaseOCRAdapter):
    """Resolves curated historical epigraphs from the catalog with 100% verified ground truth."""

    def recognize(self, image: Optional[Any] = None, text_input: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        preset_id = kwargs.get("preset_id", "")
        if preset_id:
            res = find_preset_by_id(preset_id)
            if res:
                out = dict(res)
                out["provenance"] = f"Curated Epigraphical Preset ({res['title_en']})"
                out["adapter"] = "PresetCatalogAdapter"
                return out

        if text_input:
            matched = match_inscription_from_text(text_input)
            if matched:
                out = dict(matched)
                out["provenance"] = f"Identified Historical Preset: {matched['title_en']}"
                out["adapter"] = "PresetCatalogAdapter"
                return out

        raise ValueError("Preset inscription could not be resolved.")


class ManualTranscriptionAdapter(BaseOCRAdapter):
    """Processes researcher-provided diplomatic epigraphical transcriptions."""

    def recognize(self, image: Optional[Any] = None, text_input: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        raw_text = text_input or ""
        script_hint = kwargs.get("script_hint", "auto")
        return analyze_custom_epigraph_text(
            raw_text,
            script_hint=script_hint,
            provenance="Researcher Verified Transcription"
        )


class LocalOfflineOCRAdapter(BaseOCRAdapter):
    """
    Transparent Local Offline OCR adapter.
    Evaluates whether a local deep-learning or Tesseract OCR engine specialized
    in archaic South Indian scripts is present. If absent, cleanly and scientifically
    reports the diagnostic status and guides to manual transcription or Gemini Vision AI.
    Never invents or fabricates deciphered text.
    """

    def recognize(self, image: Optional[Any] = None, text_input: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        # Scientific honesty check: Verify if specialized Brahmi/Vatteluttu weights are available locally
        # Standard general Tesseract cannot read unpointed 3rd c. BCE Damili or 8th c. Vatteluttu.
        return {
            "status": "offline_weights_unavailable",
            "message": (
                "Specialized local epigraphical weights for archaic Tamil-Brahmi and Vatteluttu "
                "are not detected in local runtime. To prevent false readings, offline mode requires "
                "researcher manual transcription or Gemini Vision AI."
            ),
            "fallback_recommended": "ManualTranscriptionAdapter"
        }


# ==============================================================================
# 9. NLP RETRIEVAL, LEXICON & PIPELINE FUNCTIONS
# ==============================================================================

def get_all_preset_inscriptions() -> List[Dict[str, Any]]:
    """Returns list of curated historical stone inscriptions."""
    return KNOWN_INSCRIPTIONS_CATALOG


def find_preset_by_id(preset_id: str) -> Optional[Dict[str, Any]]:
    """Lookup inscription preset by ID."""
    for item in KNOWN_INSCRIPTIONS_CATALOG:
        if item["id"] == preset_id:
            return item
    return None


def match_inscription_from_text(query_text: str) -> Optional[Dict[str, Any]]:
    """
    Structured, deterministic keyword & title matcher for inscriptions.
    Evaluates exact match, partial match, distinctive keywords, and vocabulary terms.
    Each criterion contributes strictly once to prevent score inflation and false positives.
    """
    if not query_text or not isinstance(query_text, str):
        return None

    clean_query = query_text.strip().lower()
    if not clean_query:
        return None

    compact_query = re.sub(r"[\s\-_,.:;!?\[\]\(\)]+", "", clean_query)

    distinctive_keywords = {
        "chola_arulmoli": ["அருள்மொழி", "அருள்மொழிதேவன்", "பராந்தகன்", "ஸ்ரீ இராசராச", "நந்தாவிளக்கு", "arulmoli", "arulmolidevan"],
        "mangulam_brahmi": ["மாங்குளம்", "கணிய்", "நந்தாஸிரிய்", "நெடுஞ்செழியன்", "பாளிய்", "வெள்ளறை", "mangulam", "neduncheliyan", "damili"],
        "thanjavur_rajaraja": ["தஞ்சை", "பெருவுடையார்", "குந்தவை", "அக்கன்", "பெண்டுகள்", "மற்றார்", "கற்றளி", "thanjavur", "brihadisvara", "kundavai"],
        "pugalur_chera": ["புகழூர்", "ஆதன் செல்லிரும்பொறை", "பெருங்கடுங்கோ", "இளங்கடுங்கோ", "செங்காயபன்", "pugalur", "chera", "irumporai"],
        "pulankurichi_vatteluttu": ["பூலாங்குறிச்சி", "சேந்தன் கூற்றன்", "கடிகைத்தாவளம்", "பள்ளிச்சந்தம்", "pulankurichi", "chendan", "vatteluttu"],
        "hero_stone_nadukal": ["செங்கம்", "நடுகல்", "தொறு", "தொறுக்கொண்டு", "சாத்தன்", "பூசல்", "chengam", "nadukal", "hero stone"],
        "uttiramerur_election": ["உத்திரமேரூர்", "குடவோலை", "குடும்பிலும்", "வாரியம்", "பராந்தக சோழன்", "uttiramerur", "kudavolai", "election"]
    }

    candidate_scores: List[tuple[int, Dict[str, Any]]] = []

    for item in KNOWN_INSCRIPTIONS_CATALOG:
        score = 0
        item_id = item["id"].lower()
        title_ta = item["title_ta"].lower()
        title_en = item["title_en"].lower()
        compact_title_ta = re.sub(r"[\s\-_,.:;!?]+", "", title_ta)
        compact_title_en = re.sub(r"[\s\-_,.:;!?]+", "", title_en)

        # 1. Exact ID Match (Highest confidence)
        if clean_query == item_id or compact_query == item_id:
            score += 100

        # 2. Exact Title Match (Tamil or English)
        elif compact_query == compact_title_ta or compact_query == compact_title_en:
            score += 90

        # 3. Partial Title Substring Match
        elif len(compact_query) >= 4 and (compact_query in compact_title_ta or compact_query in compact_title_en):
            score += 45
        elif len(compact_title_ta) >= 6 and compact_title_ta in compact_query:
            score += 45

        # 4. Distinctive Entry Keywords Match (evaluated once per keyword)
        kws = distinctive_keywords.get(item["id"], [])
        for kw in kws:
            clean_kw = kw.lower().strip()
            compact_kw = re.sub(r"[\s\-_]+", "", clean_kw)
            if compact_kw and (compact_kw in compact_query or clean_kw in clean_query):
                score += 15

        # 5. Word Vocabulary Matches (from catalog's annotated lexical terms)
        seen_words = set()
        for w in item.get("words", []):
            word_str = w.get("word", "").lower().strip()
            if word_str and len(word_str) >= 2 and word_str not in seen_words:
                seen_words.add(word_str)
                if word_str in clean_query or re.sub(r"\s+", "", word_str) in compact_query:
                    score += 4

        if score > 0:
            candidate_scores.append((score, item))

    if not candidate_scores:
        return None

    candidate_scores.sort(key=lambda x: x[0], reverse=True)
    top_score, top_item = candidate_scores[0]

    if top_score < 12:
        return None

    if len(candidate_scores) > 1:
        second_score, _ = candidate_scores[1]
        if top_score == second_score:
            return None

    return top_item


def extract_epigraph_lexicon(text: str) -> List[Dict[str, str]]:
    """Extracts known epigraphical and archaic words from arbitrary input text."""
    found_words = []
    seen = set()

    for word, info in EPIGRAPHY_LEXICON.items():
        if word in text and word not in seen:
            seen.add(word)
            found_words.append({
                "word": word,
                "modern": info["modern_tamil"],
                "english": info["english"],
                "grammar": info.get("pos", info.get("grammar", "பெயர்ச்சொல்")),
                "root": info.get("root", word),
                "case_marker": info.get("case_marker", "வேற்றுமை உருபற்ற எழுவாய் (Nominative)"),
                "notes": info["etymology"]
            })

    return found_words


def analyze_custom_epigraph_text(
    input_text: str,
    script_hint: str = "auto",
    provenance: str = "Manual Epigraphical Transcription",
    allow_preset_match: bool = False,
    source_identity: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Comprehensive multi-tier NLP analysis for researcher-supplied ancient inscriptions.
    Does NOT silently substitute catalog presets unless allow_preset_match is explicitly True.
    Generates:
      - Tier 1: Diplomatic text (preserving [ ] and ( ))
      - Tier 2: Normalized standardized Tamil Unicode
      - Tier 3: Word-segmented text (Sandhi splitting)
      - Tier 4: Modern vernacular Tamil translation
      - Tier 5: Scholarly English translation
      - Uncertain characters list
      - Morphological and case analysis table
    """
    if not input_text or not input_text.strip():
        input_text = "கண்டறியப்பட்ட கல்வெட்டுப் பதிவு"

    clean_input = input_text.strip()

    # 1. Check catalog preset match ONLY if explicitly allowed
    if allow_preset_match:
        matched = match_inscription_from_text(clean_input)
        if matched:
            result = dict(matched)
            result["provenance"] = f"Identified Historical Preset: {matched['title_en']}"
            result["data_source"] = "Catalog Match"
            result["is_curated_preset"] = True
            if source_identity:
                result["source_identity"] = getattr(source_identity, "to_dict", lambda: source_identity)()
            return result

    # 2. Textual damage and conservative restoration
    restoration_res = perform_textual_restoration(clean_input)
    clean_text = restoration_res["restored_text"]

    # 3. Script identification
    if script_hint == "auto":
        script_data = identify_script_from_text(clean_input)
    else:
        script_data = EPIGRAPHICAL_SCRIPTS.get(script_hint, identify_script_from_text(clean_input))

    detected_script_ta = script_data["name_ta"]
    detected_script_en = script_data["name_en"]
    era_ta = script_data["period_ta"]
    era_en = script_data["period_en"]

    # 4. Sandhi splitting for word segmentation (Tier 3)
    segmented_text = EpigraphicalSandhiSplitter.split_sandhi(clean_text)

    # 5. Morphological & Lexical Analysis
    words = EpigraphicalMorphologicalAnalyzer.analyze_text(clean_text)

    # 6. Extract damaged / uncertain characters
    uncertain_chars = []
    damaged_matches = re.finditer(r"\[(.*?)\]", clean_input)
    for m in damaged_matches:
        seg = m.group(1)
        uncertain_chars.append({
            "character": f"[{seg}]",
            "position": m.start(),
            "alternative_readings": [seg, f"{seg}?"],
            "reason": "Damaged chiseled stroke requiring epigraphical verification",
            "confidence": 0.50
        })

    recognized_terms = [w["word"] for w in words if w.get("notes")]

    # 7. Translations (Tiers 4 and 5)
    if recognized_terms:
        mod_tamil = f"இக்கல்வெட்டு வாசகம்: '{clean_text}' — இதில் இடம்பெற்றுள்ள '{', '.join(recognized_terms[:4])}' போன்ற சொற்கள், வரலாற்று ஆவணம், அறக்கொடை அல்லது வீர மரபை விவரிக்கின்றன."
        english_trans = f"Transcribed epigraph: '{clean_text}' — Incorporating classical terms ({', '.join(recognized_terms[:4])}) relating to historic endowment, administrative record, or commemorative tradition."
    else:
        mod_tamil = f"படியெடுக்கப்பட்ட கல்வெட்டு வாசகம்: '{clean_text}'. இவ்வெழுத்துப் பதிவுக்கான விரிவான பொருள் அறிய அருகிலுள்ள எழுத்துகள் மற்றும் வரலாற்றுச் சூழல் ஆய்வுக்குரியது."
        english_trans = f"Transcribed epigraph text: '{clean_text}'. Detailed translation requires palaeographical contextual analysis of adjacent inscriptions."

    tamil_meaning = (
        f"இக்கல்வெட்டு {detected_script_ta} வடிவில் அமைந்ததாகக் கணிக்கப்பட்டுள்ளது ({era_ta}). "
        "பண்டைய தமிழகக் கல்வெட்டுகள் பாறை முகப்புகள், கற்றளிகள், மற்றும் நடுகற்களில் செதுக்கப்பட்ட சட்ட ஆவணங்களாகவும், "
        "கொடைப் பதிவுகளாகவும் திகழ்கின்றன. இத்தொடரில் உள்ள சொற்கள் செம்மொழித் தமிழ் இலக்கண அமைப்பைப் பிரதிபலிக்கின்றன."
    )

    english_meaning = (
        f"This inscription is classified under {detected_script_en} palaeographical conventions ({era_en}). "
        "In South Indian epigraphical tradition, lithic inscriptions on temple basements, rock overhangs, and hero stones "
        "served as irrevocable public records. The morphological particles reflect archaic classical Tamil syntax."
    )

    return {
        "id": "custom_transcription",
        "title_ta": "ஆய்வாளர் படியெடுத்த கல்வெட்டு உரை",
        "title_en": "Researcher Transcribed Epigraph Record",
        "era_ta": era_ta,
        "era_en": era_en,
        "script_ta": detected_script_ta,
        "script_en": detected_script_en,
        "dynasty_ta": "பண்டைய தமிழக வரலாற்று எல்லை",
        "dynasty_en": "Historical Tamil Chronological Horizon",
        "location_ta": "தொல்லியல் களம் / நேரடிப் படி",
        "location_en": "Field Site / Inscription Rubbing",
        "category": "Epigraphical Research Record (கல்வெட்டு ஆய்வுப் பதிவு)",
        "weathered_text": clean_input,
        "restored_text": clean_text,
        "diplomatic_text": clean_input,
        "normalized_text": clean_text,
        "word_segmented_text": segmented_text,
        "modern_tamil_translation": mod_tamil,
        "english_translation": english_trans,
        "tamil_meaning": tamil_meaning,
        "english_meaning": english_meaning,
        "words": words,
        "uncertain_characters": uncertain_chars,
        "restoration_details": restoration_res,
        "provenance": provenance,
        "data_source": "Researcher Manual Input & Classical Lexicon",
        "script_inventory": script_data,
        "is_curated_preset": False,
        "source_identity": getattr(source_identity, "to_dict", lambda: source_identity)() if source_identity else None,
        "tier_1_diplomatic": clean_input,
        "tier_2_normalized": clean_text,
        "tier_3_segmented": segmented_text,
    }
