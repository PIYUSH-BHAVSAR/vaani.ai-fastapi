from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict
import re
import json
import os
from datetime import datetime
import asyncio
import aiofiles
import uuid
import yt_dlp
import webvtt
# Optional imports (install with: pip install youtube-transcript-api spacy)
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    print("youtube-transcript-api not installed. Using mock data.")

try:
    import spacy
    # Download with: python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    print("spaCy not available. Using rule-based conversion.")

app = FastAPI(title="Sign Language Translation API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("static/signs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Serve static files (for sign animations)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class YouTubeRequest(BaseModel):
    url: str
    language: Optional[str] = "en"

class Caption(BaseModel):
    start: float
    duration: float
    text: str
    sign_gloss: str
    sign_animation: Optional[str] = None
    confidence: Optional[float] = None

class TranslationResponse(BaseModel):
    video_id: str
    title: Optional[str] = None
    captions: List[Caption]
    total_duration: float
    processing_time: float

class SignLibraryItem(BaseModel):
    word: str
    gloss: str
    animation_url: str
    category: str
    difficulty: int

# Sign Language Database (expandable)
SIGN_LIBRARY = {
    # Basic words
    "hello": {"gloss": "HELLO", "animation": "hello.gif", "category": "greetings"},
    "goodbye": {"gloss": "GOODBYE", "animation": "goodbye.gif", "category": "greetings"},
    "thank": {"gloss": "THANK", "animation": "thank.gif", "category": "courtesy"},
    "you": {"gloss": "YOU", "animation": "you.gif", "category": "pronouns"},
    "i": {"gloss": "I", "animation": "i.gif", "category": "pronouns"},
    "me": {"gloss": "ME", "animation": "me.gif", "category": "pronouns"},
    "we": {"gloss": "WE", "animation": "we.gif", "category": "pronouns"},
    "they": {"gloss": "THEY", "animation": "they.gif", "category": "pronouns"},
    
    # Actions
    "go": {"gloss": "GO", "animation": "go.gif", "category": "actions"},
    "come": {"gloss": "COME", "animation": "come.gif", "category": "actions"},
    "see": {"gloss": "SEE", "animation": "see.gif", "category": "actions"},
    "hear": {"gloss": "HEAR", "animation": "hear.gif", "category": "actions"},
    "eat": {"gloss": "EAT", "animation": "eat.gif", "category": "actions"},
    "drink": {"gloss": "DRINK", "animation": "drink.gif", "category": "actions"},
    "sleep": {"gloss": "SLEEP", "animation": "sleep.gif", "category": "actions"},
    "work": {"gloss": "WORK", "animation": "work.gif", "category": "actions"},
    "play": {"gloss": "PLAY", "animation": "play.gif", "category": "actions"},
    "learn": {"gloss": "LEARN", "animation": "learn.gif", "category": "actions"},
    "teach": {"gloss": "TEACH", "animation": "teach.gif", "category": "actions"},
    "help": {"gloss": "HELP", "animation": "help.gif", "category": "actions"},
    
    # Time
    "today": {"gloss": "TODAY", "animation": "today.gif", "category": "time"},
    "tomorrow": {"gloss": "TOMORROW", "animation": "tomorrow.gif", "category": "time"},
    "yesterday": {"gloss": "YESTERDAY", "animation": "yesterday.gif", "category": "time"},
    "now": {"gloss": "NOW", "animation": "now.gif", "category": "time"},
    "later": {"gloss": "LATER", "animation": "later.gif", "category": "time"},
    "morning": {"gloss": "MORNING", "animation": "morning.gif", "category": "time"},
    "afternoon": {"gloss": "AFTERNOON", "animation": "afternoon.gif", "category": "time"},
    "evening": {"gloss": "EVENING", "animation": "evening.gif", "category": "time"},
    "night": {"gloss": "NIGHT", "animation": "night.gif", "category": "time"},
    
    # Places
    "home": {"gloss": "HOME", "animation": "home.gif", "category": "places"},
    "school": {"gloss": "SCHOOL", "animation": "school.gif", "category": "places"},
    "work": {"gloss": "WORK", "animation": "work.gif", "category": "places"},
    "hospital": {"gloss": "HOSPITAL", "animation": "hospital.gif", "category": "places"},
    "store": {"gloss": "STORE", "animation": "store.gif", "category": "places"},
    "restaurant": {"gloss": "RESTAURANT", "animation": "restaurant.gif", "category": "places"},
    
    # Family
    "family": {"gloss": "FAMILY", "animation": "family.gif", "category": "family"},
    "mother": {"gloss": "MOTHER", "animation": "mother.gif", "category": "family"},
    "father": {"gloss": "FATHER", "animation": "father.gif", "category": "family"},
    "brother": {"gloss": "BROTHER", "animation": "brother.gif", "category": "family"},
    "sister": {"gloss": "SISTER", "animation": "sister.gif", "category": "family"},
    "child": {"gloss": "CHILD", "animation": "child.gif", "category": "family"},
    
    # Common words
    "good": {"gloss": "GOOD", "animation": "good.gif", "category": "adjectives"},
    "bad": {"gloss": "BAD", "animation": "bad.gif", "category": "adjectives"},
    "big": {"gloss": "BIG", "animation": "big.gif", "category": "adjectives"},
    "small": {"gloss": "SMALL", "animation": "small.gif", "category": "adjectives"},
    "important": {"gloss": "IMPORTANT", "animation": "important.gif", "category": "adjectives"},
    "beautiful": {"gloss": "BEAUTIFUL", "animation": "beautiful.gif", "category": "adjectives"},
    "happy": {"gloss": "HAPPY", "animation": "happy.gif", "category": "emotions"},
    "sad": {"gloss": "SAD", "animation": "sad.gif", "category": "emotions"},
    "love": {"gloss": "LOVE", "animation": "love.gif", "category": "emotions"},
    
    # Questions
    "what": {"gloss": "WHAT", "animation": "what.gif", "category": "questions"},
    "who": {"gloss": "WHO", "animation": "who.gif", "category": "questions"},
    "where": {"gloss": "WHERE", "animation": "where.gif", "category": "questions"},
    "when": {"gloss": "WHEN", "animation": "when.gif", "category": "questions"},
    "why": {"gloss": "WHY", "animation": "why.gif", "category": "questions"},
    "how": {"gloss": "HOW", "animation": "how.gif", "category": "questions"},
    
    # Common phrases
    "please": {"gloss": "PLEASE", "animation": "please.gif", "category": "courtesy"},
    "sorry": {"gloss": "SORRY", "animation": "sorry.gif", "category": "courtesy"},
    "welcome": {"gloss": "WELCOME", "animation": "welcome.gif", "category": "courtesy"},
    "excuse": {"gloss": "EXCUSE", "animation": "excuse.gif", "category": "courtesy"},
    "yes": {"gloss": "YES", "animation": "yes.gif", "category": "responses"},
    "no": {"gloss": "NO", "animation": "no.gif", "category": "responses"},
    "maybe": {"gloss": "MAYBE", "animation": "maybe.gif", "category": "responses"},
    "understand": {"gloss": "UNDERSTAND", "animation": "understand.gif", "category": "communication"},
    "know": {"gloss": "KNOW", "animation": "know.gif", "category": "communication"},
    "think": {"gloss": "THINK", "animation": "think.gif", "category": "communication"},
    "feel": {"gloss": "FEEL", "animation": "feel.gif", "category": "communication"},
    "want": {"gloss": "WANT", "animation": "want.gif", "category": "communication"},
    "need": {"gloss": "NEED", "animation": "need.gif", "category": "communication"},
    "like": {"gloss": "LIKE", "animation": "like.gif", "category": "communication"},
    "have": {"gloss": "HAVE", "animation": "have.gif", "category": "communication"},
    "get": {"gloss": "GET", "animation": "get.gif", "category": "communication"},
    "give": {"gloss": "GIVE", "animation": "give.gif", "category": "communication"},
    "take": {"gloss": "TAKE", "animation": "take.gif", "category": "communication"},
    "make": {"gloss": "MAKE", "animation": "make.gif", "category": "communication"},
    "do": {"gloss": "DO", "animation": "do.gif", "category": "communication"},
    "can": {"gloss": "CAN", "animation": "can.gif", "category": "modal"},
    "will": {"gloss": "WILL", "animation": "will.gif", "category": "modal"},
    "should": {"gloss": "SHOULD", "animation": "should.gif", "category": "modal"},
    "must": {"gloss": "MUST", "animation": "must.gif", "category": "modal"},
}
def extract_captions_with_ytdlp(video_url: str, lang: str = "en") -> List[Dict]:
    temp_id = str(uuid.uuid4())
    output_template = f"/tmp/{temp_id}.%(ext)s"

    ydl_opts = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "skip_download": True,
        "outtmpl": output_template,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_url])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"yt-dlp error: {str(e)}")

    vtt_file = f"/tmp/{temp_id}.{lang}.vtt"
    if not os.path.exists(vtt_file):
        raise HTTPException(status_code=404, detail="Captions not available.")

    captions = []
    for caption in webvtt.read(vtt_file):
        captions.append({
            "start": convert_timestamp(caption.start),
            "duration": convert_duration(caption.start, caption.end),
            "text": caption.text.strip().replace('\n', ' ')
        })

    os.remove(vtt_file)
    return captions

def convert_timestamp(ts: str) -> float:
    h, m, s = ts.replace(',', '.').split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)

def convert_duration(start: str, end: str) -> float:
    return convert_timestamp(end) - convert_timestamp(start)

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    raise ValueError("Invalid YouTube URL")

def get_mock_captions(video_id: str) -> List[Dict]:
    """Generate mock captions for testing when API is not available"""
    mock_data = [
        {"start": 0.0, "duration": 3.5, "text": "Hello everyone, welcome to our channel"},
        {"start": 3.5, "duration": 4.0, "text": "Today we will learn about sign language"},
        {"start": 7.5, "duration": 3.0, "text": "This is very important for communication"},
        {"start": 10.5, "duration": 4.5, "text": "We will start with basic greetings and common words"},
        {"start": 15.0, "duration": 3.0, "text": "Please follow along and practice with us"},
        {"start": 18.0, "duration": 2.5, "text": "Thank you for watching our video"},
        {"start": 20.5, "duration": 3.0, "text": "Don't forget to like and subscribe"},
        {"start": 23.5, "duration": 2.0, "text": "See you in the next lesson"},
    ]
    return mock_data
def rule_based_text_to_gloss(text: str) -> str:
    """Convert English text to ASL gloss using improved rule-based logic."""
    import re

    # Normalize and clean input
    text = text.upper()
    text = re.sub(r"[^\w\s']", '', text)  # Keep apostrophes for contractions
    words = text.split()

    gloss_words = []
    time_markers = []
    question_words = []
    
    skip_words = {
        'THE', 'A', 'AN', 'IS', 'ARE', 'WAS', 'WERE', 'BE', 'BEEN', 'BEING',
        'TO', 'OF', 'FOR', 'WITH', 'BY', 'FROM', 'AT', 'IN', 'ON', 'THAT',
        'THIS', 'THESE', 'THOSE', 'DOES', 'DID', 'HAS', 'HAVE', 'HAD', 'AND'
    }

    transformations = {
        'AM': 'I', 'MY': 'ME', 'YOUR': 'YOU', 'HIS': 'HE', 'HER': 'SHE',
        'OUR': 'WE', 'THEIR': 'THEY',
        'GOING': 'GO', 'CAME': 'COME', 'COMING': 'COME',
        'SEEN': 'SEE', 'SEEING': 'SEE', 'SAW': 'SEE',
        'HEARD': 'HEAR', 'HEARING': 'HEAR',
        'EATING': 'EAT', 'ATE': 'EAT', 'DRINKING': 'DRINK', 'DRANK': 'DRINK',
        'SLEEPING': 'SLEEP', 'WORKING': 'WORK', 'PLAYING': 'PLAY',
        'LEARNING': 'LEARN', 'TAUGHT': 'TEACH', 'TEACHING': 'TEACH',
        'HELPING': 'HELP', 'WATCHING': 'WATCH', 'THINKING': 'THINK',
        'FEELING': 'FEEL', 'LOVING': 'LOVE', 'LIKING': 'LIKE',
        'KNOWING': 'KNOW', 'NEEDING': 'NEED', 'WANTING': 'WANT',
        'MAKING': 'MAKE', 'GIVING': 'GIVE', 'TAKING': 'TAKE',
        'HAVING': 'HAVE', 'DOING': 'DO', 'BEING': 'BE',
        'DONT': 'NOT', "DON'T": 'NOT', 'NOTHING': 'NOTHING',
        'CANT': 'CANNOT', "CAN'T": 'CANNOT',
        'WONT': 'WILL NOT', "WON'T": 'WILL NOT',
        'SHOULD': '', 'COULD': '', 'WOULD': '',
    }

    question_set = {'WHAT', 'WHO', 'WHERE', 'WHEN', 'WHY', 'HOW'}
    time_words = {'TODAY', 'TOMORROW', 'YESTERDAY', 'NOW', 'LATER', 'MORNING', 'AFTERNOON', 'EVENING', 'NIGHT'}

    for word in words:
        word = transformations.get(word, word)

        # Skip known filler words
        if word in skip_words:
            continue

        # Extract time words separately
        if word in time_words:
            time_markers.append(word)
            continue

        # Extract question words separately
        if word in question_set:
            question_words.append(word)
            continue

        # Handle compound substitutions like "WILL NOT"
        if " " in word:
            for w in word.split():
                if w not in skip_words:
                    gloss_words.append(w)
        else:
            gloss_words.append(word)

    # Final assembly: QUESTION + TIME + REST
    final_gloss = question_words + time_markers + gloss_words
    return ' '.join(final_gloss)



def spacy_text_to_gloss(text: str) -> str:
    """Convert English text to ASL gloss using spaCy NLP"""
    if not SPACY_AVAILABLE:
        return rule_based_text_to_gloss(text)
    
    doc = nlp(text)
    gloss_words = []
    
    # Extract main content words
    for token in doc:
        # Skip punctuation, spaces, and stop words
        if token.is_punct or token.is_space or token.is_stop:
            continue
        
        # Use lemma for base form
        word = token.lemma_.upper()
        
        # Handle pronouns
        if token.pos_ == "PRON":
            if word in ["I", "ME", "MY", "MINE"]:
                word = "I"
            elif word in ["YOU", "YOUR", "YOURS"]:
                word = "YOU"
            elif word in ["HE", "HIM", "HIS"]:
                word = "HE"
            elif word in ["SHE", "HER", "HERS"]:
                word = "SHE"
            elif word in ["WE", "US", "OUR", "OURS"]:
                word = "WE"
            elif word in ["THEY", "THEM", "THEIR", "THEIRS"]:
                word = "THEY"
        
        # Handle verbs (use base form)
        elif token.pos_ == "VERB":
            word = token.lemma_.upper()
        
        # Handle questions (move to front)
        if word in ["WHAT", "WHO", "WHERE", "WHEN", "WHY", "HOW"]:
            if gloss_words and gloss_words[0] not in ["WHAT", "WHO", "WHERE", "WHEN", "WHY", "HOW"]:
                gloss_words.insert(0, word)
                continue
        
        gloss_words.append(word)
    
    return ' '.join(gloss_words)

def get_sign_animation(word: str) -> Optional[str]:
    """Get sign animation URL for a word"""
    word_lower = word.lower()
    if word_lower in SIGN_LIBRARY:
        return f"/static/signs/{SIGN_LIBRARY[word_lower]['animation']}"
    return None

def process_captions_to_signs(captions: List[Dict]) -> List[Caption]:
    """Process captions and convert to sign language"""
    sign_captions = []
    
    for caption in captions:
        text = caption['text']
        
        # Convert to gloss
        if SPACY_AVAILABLE:
            gloss = spacy_text_to_gloss(text)
        else:
            gloss = rule_based_text_to_gloss(text)
        
        # Find sign animation
        words = gloss.split()
        animation_url = None
        
        # Try to find animation for the first recognizable word
        for word in words:
            animation = get_sign_animation(word)
            if animation:
                animation_url = animation
                break
        
        # If no specific animation found, use a general one
        if not animation_url:
            animation_url = "/static/signs/general.gif"
        
        sign_caption = Caption(
            start=caption['start'],
            duration=caption['duration'],
            text=text,
            sign_gloss=gloss,
            sign_animation=animation_url,
            confidence=0.9  # Mock confidence score
        )
        
        sign_captions.append(sign_caption)
    
    return sign_captions

@app.get("/")
async def root():
    return {"message": "Sign Language Translation API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "youtube_api": YOUTUBE_API_AVAILABLE,
        "spacy": SPACY_AVAILABLE,
        "sign_library_size": len(SIGN_LIBRARY)
    }

@app.post("/translate", response_model=TranslationResponse)
async def translate_video(request: YouTubeRequest):
    start_time = datetime.now()

    try:
        video_id = extract_video_id(request.url)

        # Get real captions using yt-dlp
        captions = extract_captions_with_ytdlp(request.url, request.language)

        # Convert to sign language
        sign_captions = process_captions_to_signs(captions)

        total_duration = max([c.start + c.duration for c in sign_captions]) if sign_captions else 0
        processing_time = (datetime.now() - start_time).total_seconds()

        return TranslationResponse(
            video_id=video_id,
            title=f"Translated Video: {video_id}",
            captions=sign_captions,
            total_duration=total_duration,
            processing_time=processing_time
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/sign-library")
async def get_sign_library():
    """Get the complete sign language library"""
    library = []
    for word, data in SIGN_LIBRARY.items():
        library.append(SignLibraryItem(
            word=word,
            gloss=data["gloss"],
            animation_url=f"/static/signs/{data['animation']}",
            category=data["category"],
            difficulty=1  # Default difficulty
        ))
    return {"library": library, "total_signs": len(library)}

@app.get("/sign-library/{category}")
async def get_sign_library_by_category(category: str):
    """Get sign library filtered by category"""
    library = []
    for word, data in SIGN_LIBRARY.items():
        if data["category"] == category:
            library.append(SignLibraryItem(
                word=word,
                gloss=data["gloss"],
                animation_url=f"/static/signs/{data['animation']}",
                category=data["category"],
                difficulty=1
            ))
    return {"library": library, "category": category, "total_signs": len(library)}

@app.get("/categories")
async def get_categories():
    """Get all available sign categories"""
    categories = set()
    for data in SIGN_LIBRARY.values():
        categories.add(data["category"])
    return {"categories": sorted(list(categories))}

@app.post("/text-to-gloss")
async def text_to_gloss(text: str):
    gloss = rule_based_text_to_gloss(text)  # force it
    return {"text": text, "gloss": gloss}

@app.get("/video-info/{video_id}")
async def get_video_info(video_id: str):
    """Get video information (mock implementation)"""
    return {
        "video_id": video_id,
        "title": f"Video {video_id}",
        "duration": 180,  # Mock duration
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
