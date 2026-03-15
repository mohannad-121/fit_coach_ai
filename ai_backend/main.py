from __future__ import annotations

import os
import logging
import re
import json
import uuid
import shutil
from functools import lru_cache
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from ai_engine import AIEngine
from domain_router import DomainRouter
from dataset_registry import DatasetRegistry
from knowledge_engine import KnowledgeEngine
from llm_client import LLMClient
from logic_engine import evaluate_logic_metrics
from memory_system import MemorySystem
from moderation_layer import ModerationLayer
from predict import predict_goal, predict_plan_intent, predict_success
from response_datasets import ResponseDatasets
from voice.stt import WhisperSTT
from voice.tts import LocalTTS
from voice.voice_pipeline import VoicePipeline, VoicePipelineError, VoicePipelineResult
from nlp_utils import (
    extract_first_int,
    fuzzy_contains_any,
    normalize_text,
    repair_mojibake as nlp_repair_mojibake,
    repair_mojibake_deep,
)


app = FastAPI(title="AI Fitness Coach Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
STATIC_DIR = BACKEND_DIR / "static"
STATIC_AUDIO_DIR = STATIC_DIR / "audio"
STATIC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    language: Optional[str] = "en"
    stream: Optional[bool] = False
    user_profile: Optional[Dict[str, Any]] = None
    tracking_summary: Optional[Dict[str, Any]] = None
    recent_messages: Optional[list[Dict[str, Any]]] = None
    plan_snapshot: Optional[Dict[str, Any]] = None


def _repair_mojibake(text: str) -> str:
    return nlp_repair_mojibake(text)


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str
    language: str
    action: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    @field_validator("reply", mode="before")
    @classmethod
    def _normalize_reply_text(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return _repair_mojibake(value)

    @field_validator("data", mode="before")
    @classmethod
    def _normalize_data_payload(cls, value: Any) -> Any:
        return repair_mojibake_deep(value)


class VoiceChatResponse(BaseModel):
    transcript: str
    reply: str
    audio_path: str
    conversation_id: str
    language: str


class PlanActionRequest(BaseModel):
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


class GoalPredictionRequest(BaseModel):
    age: Optional[float] = 0.0
    gender: Optional[str] = "Other"
    weight_kg: Optional[float] = 0.0
    height_m: Optional[float] = None
    height_cm: Optional[float] = None
    bmi: Optional[float] = 0.0
    fat_percentage: Optional[float] = 0.0
    workout_frequency_days_week: Optional[float] = 0.0
    experience_level: Optional[float] = 0.0
    calories_burned: Optional[float] = 0.0
    avg_bpm: Optional[float] = 0.0


class SuccessPredictionRequest(BaseModel):
    age: Optional[float] = 0.0
    gender: Optional[str] = "Other"
    membership_type: Optional[str] = "Unknown"
    workout_type: Optional[str] = "Unknown"
    workout_duration_minutes: Optional[float] = 0.0
    calories_burned: Optional[float] = 0.0
    check_in_hour: Optional[int] = 0
    check_in_time: Optional[str] = None


class LogicEvaluationRequest(BaseModel):
    start_value: Optional[float] = None
    current_value: Optional[float] = None
    target_value: Optional[float] = None
    direction: str = "decrease"
    weight_history: Optional[list[float]] = None
    previous_value: Optional[float] = None
    elapsed_weeks: float = 1.0


class PlanIntentPredictionRequest(BaseModel):
    message: str


def _resolve_response_dataset_dir() -> Path:
    base_data_dir = Path(__file__).resolve().parent / "data"
    candidates = [
        base_data_dir / "week2",
        base_data_dir / "chat data",
    ]
    required_files = ("conversation_intents.json", "workout_programs.json", "nutrition_programs.json")
    for candidate in candidates:
        if all((candidate / name).exists() for name in required_files):
            return candidate
    return candidates[0]


ROUTER = DomainRouter(threshold=0.42, enable_semantic=False)
MODERATION = ModerationLayer()
LLM = LLMClient()
AI_ENGINE = AIEngine(Path(__file__).resolve().parent / "exercises.json")
NUTRITION_KB = KnowledgeEngine(Path(__file__).resolve().parent / "knowledge" / "dataforproject.txt")
RESPONSE_DATASET_DIR = _resolve_response_dataset_dir()
RESPONSE_DATASETS = ResponseDatasets(RESPONSE_DATASET_DIR)
CHAT_RESPONSE_MODE = os.getenv('CHAT_RESPONSE_MODE', 'ai_hybrid').strip().lower()
VOICE_STT = WhisperSTT(model_name=os.getenv("WHISPER_MODEL", "openai/whisper-base"))
VOICE_TTS = LocalTTS(output_dir=STATIC_AUDIO_DIR)
VOICE_PIPELINE = VoicePipeline(stt_engine=VOICE_STT, tts_engine=VOICE_TTS, llm_client=LLM)
DATASET_REGISTRY = DatasetRegistry(
    Path(r"D:\chatbot coach\Dataset"),
    Path(__file__).resolve().parent / "data" / "dataset_registry_index.json",
)
DATASET_REGISTRY.build_index(force_rebuild=False)

MEMORY_SESSIONS: Dict[str, MemorySystem] = {}
PENDING_PLANS: Dict[str, Dict[str, Any]] = {}
USER_STATE: Dict[str, Dict[str, Any]] = {}

WEEK_DAYS = [
    ("Saturday", "السبت"),
    ("Sunday", "الأحد"),
    ("Monday", "الاثنين"),
    ("Tuesday", "الثلاثاء"),
    ("Wednesday", "الأربعاء"),
    ("Thursday", "الخميس"),
    ("Friday", "الجمعة"),
]

GREETING_KEYWORDS = {
    "hi",
    "hello",
    "hey",
    "Ù…Ø±Ø­Ø¨Ø§",
    "Ø§Ù‡Ù„Ø§",
    "Ù‡Ù„Ø§",
    "Ø§Ù„Ø³Ù„Ø§Ù… Ø¹Ù„ÙŠÙƒÙ…",
}

NAME_KEYWORDS = {"name", "Ø§Ø³Ù…Ùƒ", "Ø´Ùˆ Ø§Ø³Ù…Ùƒ", "Ù…ÙŠÙ† Ø§Ù†Øª"}
HOW_ARE_YOU_KEYWORDS = {"how are you", "ÙƒÙŠÙÙƒ", "Ø´Ù„ÙˆÙ†Ùƒ", "ÙƒÙŠÙ Ø­Ø§Ù„Ùƒ"}
WORKOUT_PLAN_KEYWORDS = {
    "workout plan",
    "training plan",
    "program",
    "Ø®Ø·Ø© ØªÙ…Ø§Ø±ÙŠÙ†",
    "Ø¨Ø±Ù†Ø§Ù…Ø¬ ØªÙ…Ø§Ø±ÙŠÙ†",
    "Ø¬Ø¯ÙˆÙ„ ØªÙ…Ø§Ø±ÙŠÙ†",
}
NUTRITION_PLAN_KEYWORDS = {
    "nutrition plan",
    "meal plan",
    "diet plan",
    "Ø®Ø·Ø© ØºØ°Ø§Ø¦ÙŠØ©",
    "Ø®Ø·Ø© ØªØºØ°ÙŠØ©",
    "Ø¬Ø¯ÙˆÙ„ ÙˆØ¬Ø¨Ø§Øª",
}
NUTRITION_KB_KEYWORDS = {
    "nutrition",
    "diet",
    "meal",
    "food",
    "foods",
    "ingredient",
    "calories",
    "protein",
    "carbs",
    "fat",
    "allergy",
    "allergies",
    "diabetes",
    "blood pressure",
    "cholesterol",
    "heart disease",
    "تغذية",
    "غذاء",
    "اكل",
    "وجبة",
    "وجبات",
    "سعرات",
    "بروتين",
    "كارب",
    "دهون",
    "حساسية",
    "سكري",
    "ضغط",
    "كوليسترول",
    "قلب",
    "خطة غذائية",
    "دايت",
}
PROGRESS_KEYWORDS = {"progress", "tracking", "adherence", "Ø§Ù„Ø§Ù„ØªØ²Ø§Ù…", "Ø§Ù„ØªÙ‚Ø¯Ù…", "Ø§Ù†Ø¬Ø§Ø²"}
PERFORMANCE_ANALYSIS_KEYWORDS = {
    "performance",
    "weekly performance",
    "monthly performance",
    "performance analysis",
    "rate of progress",
    "on track",
    "ahead of schedule",
    "behind schedule",
    "weeks remaining",
    "timeline",
    "تحليل الأداء",
    "تحليل الاداء",
    "اداء",
    "أداء",
    "اسبوعي",
    "أسبوعي",
    "شهري",
    "تحليل التقدم",
    "على المسار",
    "متقدم",
    "متأخر",
    "كم أسبوع",
    "كم اسبوع",
    "قديش ضايل",
    "كم ضايل",
    "ضايلي",
    "ضايل",
    "ضايل لهدفي",
    "remaining time",
    "time to goal",
    "remaining weeks",
}
APPROVE_KEYWORDS = {"approve", "yes", "ÙˆØ§ÙÙ‚", "Ø§Ø¹ØªÙ…Ø¯", "Ù…ÙˆØ§ÙÙ‚"}
REJECT_KEYWORDS = {"reject", "no", "Ø±ÙØ¶", "Ù„Ø§", "ØºÙŠØ± Ø§Ù„Ø®Ø·Ø©", "Ø¨Ø¯Ù„ Ø§Ù„Ø®Ø·Ø©"}
JORDANIAN_HINTS = {"Ø´Ùˆ", "Ø¨Ø¯Ùƒ", "Ù‡Ù„Ø§", "Ù„Ø³Ø§", "Ù…Ø´", "ÙƒØªÙŠØ±", "Ù…Ù†ÙŠØ­", "ØªÙ…Ø§Ù…"}


PLAN_CHOICE_KEYWORDS = {
    "choose",
    "option",
    "pick",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "Ø§Ø®ØªØ§Ø±",
    "Ø®ÙŠØ§Ø±",
    "Ø§ÙˆÙ„",
    "Ø«Ø§Ù†ÙŠ",
    "Ø«Ø§Ù„Ø«",
    "Ø±Ø§Ø¨Ø¹",
    "Ø®Ø§Ù…Ø³",
}
PLAN_REFRESH_KEYWORDS = {"more options", "another options", "Ø®ÙŠØ§Ø±Ø§Øª Ø§ÙƒØ«Ø±", "Ø®ÙŠØ§Ø±Ø§Øª Ø£Ø®Ø±Ù‰", "ØºÙŠØ±Ù‡Ù…"}
APPROVE_KEYWORDS = APPROVE_KEYWORDS | {"accept", "okay", "ok", "Ù…Ø§Ø´ÙŠ"}
REJECT_KEYWORDS = REJECT_KEYWORDS | {"decline", "cancel"}
WORKOUT_PLAN_KEYWORDS = WORKOUT_PLAN_KEYWORDS | {"workout", "training", "routine", "ØªÙ…Ø§Ø±ÙŠÙ†", "Ø¨Ø±Ù†Ø§Ù…Ø¬"}
NUTRITION_PLAN_KEYWORDS = NUTRITION_PLAN_KEYWORDS | {"nutrition", "diet", "meal", "ØªØºØ°ÙŠØ©", "ÙˆØ¬Ø¨Ø§Øª"}


THANKS_KEYWORDS = {
    "thanks",
    "thank you",
    "thx",
    "good job",
    "nice",
    "awesome",
    "great",
    "well done",
    "\u0634\u0643\u0631\u0627",
    "\u064a\u0633\u0644\u0645\u0648",
    "\u064a\u0639\u0637\u064a\u0643 \u0627\u0644\u0639\u0627\u0641\u064a\u0629",
    "\u0627\u062d\u0633\u0646\u062a",
    "\u0623\u062d\u0633\u0646\u062a",
}
WHO_AM_I_KEYWORDS = {
    "who am i",
    "tell me about me",
    "my info",
    "my profile",
    "\u0645\u064a\u0646 \u0627\u0646\u0627",
    "\u0645\u064a\u0646 \u0623\u0646\u0627",
    "\u0639\u0631\u0641\u0646\u064a",
    "\u0645\u0639\u0644\u0648\u0645\u0627\u062a\u064a",
    "\u0645\u0644\u0641\u064a",
}
ASK_MY_AGE_KEYWORDS = {"my age", "how old am i", "\u0643\u0645 \u0639\u0645\u0631\u064a", "\u0639\u0645\u0631\u064a"}
ASK_MY_HEIGHT_KEYWORDS = {"my height", "how tall am i", "\u0637\u0648\u0644\u064a", "\u0643\u0645 \u0637\u0648\u0644\u064a"}
ASK_MY_WEIGHT_KEYWORDS = {"my weight", "how much do i weigh", "\u0648\u0632\u0646\u064a", "\u0643\u0645 \u0648\u0632\u0646\u064a"}
ASK_MY_GOAL_KEYWORDS = {"my goal", "what is my goal", "\u0647\u062f\u0641\u064a", "\u0634\u0648 \u0647\u062f\u0641\u064a", "\u0645\u0627 \u0647\u062f\u0641\u064a"}

PROGRESS_CONCERN_KEYWORDS = {
    "no progress",
    "no change",
    "not improving",
    "plateau",
    "stuck",
    "\u0645\u0627 \u0641\u064a \u0641\u0631\u0642",
    "\u0645\u0641\u064a\u0634 \u0641\u0631\u0642",
    "\u0645\u0627 \u062a\u063a\u064a\u0631 \u062c\u0633\u0645\u064a",
    "\u062c\u0633\u0645\u064a \u0645\u0627 \u062a\u063a\u064a\u0631",
    "\u062b\u0627\u0628\u062a",
    "\u0645\u0627 \u0639\u0645 \u0628\u0646\u0632\u0644",
    "\u0645\u0627 \u0639\u0645 \u0628\u0632\u064a\u062f",
}
TROUBLESHOOT_KEYWORDS = {
    "exercise wrong",
    "wrong form",
    "bad form",
    "pain during exercise",
    "injury",
    "hurts",
    "movement is wrong",
    "\u0627\u0644\u062a\u0645\u0631\u064a\u0646 \u063a\u0644\u0637",
    "\u062d\u0631\u0643\u062a\u064a \u063a\u0644\u0637",
    "\u0628\u0648\u062c\u0639\u0646\u064a",
    "\u064a\u0648\u062c\u0639\u0646\u064a",
    "\u0627\u0635\u0627\u0628\u0629",
    "\u0625\u0635\u0627\u0628\u0629",
    "\u0623\u0644\u0645",
    "\u0648\u062c\u0639",
}
PLAN_STATUS_KEYWORDS = {
    "active plan",
    "current plan",
    "\u0647\u0644 \u0639\u0646\u062f\u064a \u062e\u0637\u0629",
    "\u0634\u0648 \u062e\u0637\u062a\u064a",
    "\u0645\u0627 \u0647\u064a \u062e\u0637\u062a\u064a",
    "\u062e\u0637\u062a\u064a \u0627\u0644\u062d\u0627\u0644\u064a\u0629",
}

# Add robust Arabic forms to avoid encoding-related misses.
GREETING_KEYWORDS = GREETING_KEYWORDS | {
    "\u0645\u0631\u062d\u0628\u0627",
    "\u0627\u0647\u0644\u0627",
    "\u0647\u0644\u0627",
    "\u0627\u0644\u0633\u0644\u0627\u0645 \u0639\u0644\u064a\u0643\u0645",
}
NAME_KEYWORDS = NAME_KEYWORDS | {
    "\u0627\u0633\u0645\u0643",
    "\u0634\u0648 \u0627\u0633\u0645\u0643",
    "\u0645\u064a\u0646 \u0627\u0646\u062a",
}
HOW_ARE_YOU_KEYWORDS = HOW_ARE_YOU_KEYWORDS | {
    "\u0643\u064a\u0641\u0643",
    "\u0634\u0644\u0648\u0646\u0643",
    "\u0643\u064a\u0641 \u062d\u0627\u0644\u0643",
}
WORKOUT_PLAN_KEYWORDS = WORKOUT_PLAN_KEYWORDS | {
    "\u062e\u0637\u0629 \u062a\u0645\u0627\u0631\u064a\u0646",
    "\u0628\u0631\u0646\u0627\u0645\u062c \u062a\u0645\u0627\u0631\u064a\u0646",
    "\u062c\u062f\u0648\u0644 \u062a\u0645\u0627\u0631\u064a\u0646",
    "\u062a\u0645\u0631\u064a\u0646",
    "\u062a\u0645\u0627\u0631\u064a\u0646",
    "\u0627\u0644\u0635\u062f\u0631",
    "\u0627\u0644\u0638\u0647\u0631",
    "\u0627\u0644\u0633\u0627\u0642",
    "\u0627\u0644\u0627\u0631\u062c\u0644",
    "\u0627\u0644\u0643\u062a\u0641",
}
NUTRITION_PLAN_KEYWORDS = NUTRITION_PLAN_KEYWORDS | {
    "\u062e\u0637\u0629 \u063a\u0630\u0627\u0626\u064a\u0629",
    "\u062e\u0637\u0629 \u062a\u063a\u0630\u064a\u0629",
    "\u062c\u062f\u0648\u0644 \u0648\u062c\u0628\u0627\u062a",
    "\u0633\u0639\u0631\u0627\u062a",
    "\u0628\u0631\u0648\u062a\u064a\u0646",
}
PROGRESS_KEYWORDS = PROGRESS_KEYWORDS | {
    "\u0627\u0644\u062a\u0632\u0627\u0645",
    "\u0627\u0644\u062a\u0642\u062f\u0645",
    "\u0627\u0646\u062c\u0627\u0632",
    "\u0645\u0627 \u0641\u064a \u0641\u0631\u0642",
}
JORDANIAN_HINTS = JORDANIAN_HINTS | {
    "\u0634\u0648",
    "\u0628\u062f\u0643",
    "\u0645\u0634",
    "\u0645\u0646\u064a\u062d",
    "\u062a\u0645\u0627\u0645",
}

STRONG_DOMAIN_KEYWORDS = {
    "workout",
    "exercise",
    "training",
    "gym",
    "muscle",
    "strength",
    "hypertrophy",
    "progressive overload",
    "overload",
    "sets",
    "reps",
    "rest time",
    "nutrition",
    "meal",
    "diet",
    "calories",
    "protein",
    "\u062a\u0645\u0631\u064a\u0646",
    "\u062a\u0645\u0627\u0631\u064a\u0646",
    "\u062a\u062f\u0631\u064a\u0628",
    "\u0627\u0644\u0635\u062f\u0631",
    "\u0639\u0636\u0644",
    "\u0639\u0636\u0644\u0627\u062a",
    "\u0642\u0648\u0629",
    "\u0636\u062e\u0627\u0645\u0629",
    "\u062d\u0645\u0644 \u062a\u062f\u0631\u064a\u062c\u064a",
    "\u0627\u0648\u0641\u0631\u0644\u0648\u062f",
    "\u0645\u062c\u0645\u0648\u0639\u0627\u062a",
    "\u062a\u0643\u0631\u0627\u0631\u0627\u062a",
    "\u063a\u0630\u0627\u0621",
    "\u062a\u063a\u0630\u064a\u0629",
    "\u0648\u062c\u0628\u0627\u062a",
    "\u0633\u0639\u0631\u0627\u062a",
    "\u0628\u0631\u0648\u062a\u064a\u0646",
    "\u0644\u064a\u0627\u0642\u0629",
}

ML_GOAL_QUERY_KEYWORDS = {
    "predict goal",
    "goal prediction",
    "predict my goal",
    "best goal for me",
    "recommended goal",
    "what goal suits me",
    "توقع الهدف",
    "تنبؤ الهدف",
    "شو الهدف المناسب",
    "اي هدف مناسب",
    "ما الهدف المناسب",
    "توقع هدفي",
}

ML_SUCCESS_QUERY_KEYWORDS = {
    "success prediction",
    "predict success",
    "success probability",
    "chance of success",
    "will i succeed",
    "am i likely to succeed",
    "نسبة النجاح",
    "احتمال النجاح",
    "توقع النجاح",
    "هل رح انجح",
    "هل سأنجح",
    "هل رح ألتزم",
    "هل سانجح",
}

ML_GENERAL_PREDICTION_KEYWORDS = {
    "predict",
    "prediction",
    "ai prediction",
    "model prediction",
    "توقع",
    "تنبؤ",
    "توقعي",
}


def _expand_keyword_set_with_repair(values: set[str]) -> set[str]:
    expanded = set(values)
    for value in list(values):
        repaired = _repair_mojibake(value)
        if repaired:
            expanded.add(repaired)
    return expanded


GREETING_KEYWORDS = _expand_keyword_set_with_repair(GREETING_KEYWORDS)
NAME_KEYWORDS = _expand_keyword_set_with_repair(NAME_KEYWORDS)
HOW_ARE_YOU_KEYWORDS = _expand_keyword_set_with_repair(HOW_ARE_YOU_KEYWORDS)
WORKOUT_PLAN_KEYWORDS = _expand_keyword_set_with_repair(WORKOUT_PLAN_KEYWORDS)
NUTRITION_PLAN_KEYWORDS = _expand_keyword_set_with_repair(NUTRITION_PLAN_KEYWORDS)
NUTRITION_KB_KEYWORDS = _expand_keyword_set_with_repair(NUTRITION_KB_KEYWORDS)
PROGRESS_KEYWORDS = _expand_keyword_set_with_repair(PROGRESS_KEYWORDS)
APPROVE_KEYWORDS = _expand_keyword_set_with_repair(APPROVE_KEYWORDS)
REJECT_KEYWORDS = _expand_keyword_set_with_repair(REJECT_KEYWORDS)
JORDANIAN_HINTS = _expand_keyword_set_with_repair(JORDANIAN_HINTS)
PLAN_CHOICE_KEYWORDS = _expand_keyword_set_with_repair(PLAN_CHOICE_KEYWORDS)
PLAN_REFRESH_KEYWORDS = _expand_keyword_set_with_repair(PLAN_REFRESH_KEYWORDS)
THANKS_KEYWORDS = _expand_keyword_set_with_repair(THANKS_KEYWORDS)
WHO_AM_I_KEYWORDS = _expand_keyword_set_with_repair(WHO_AM_I_KEYWORDS)
ASK_MY_AGE_KEYWORDS = _expand_keyword_set_with_repair(ASK_MY_AGE_KEYWORDS)
ASK_MY_HEIGHT_KEYWORDS = _expand_keyword_set_with_repair(ASK_MY_HEIGHT_KEYWORDS)
ASK_MY_WEIGHT_KEYWORDS = _expand_keyword_set_with_repair(ASK_MY_WEIGHT_KEYWORDS)
ASK_MY_GOAL_KEYWORDS = _expand_keyword_set_with_repair(ASK_MY_GOAL_KEYWORDS)
PROGRESS_CONCERN_KEYWORDS = _expand_keyword_set_with_repair(PROGRESS_CONCERN_KEYWORDS)
TROUBLESHOOT_KEYWORDS = _expand_keyword_set_with_repair(TROUBLESHOOT_KEYWORDS)
PLAN_STATUS_KEYWORDS = _expand_keyword_set_with_repair(PLAN_STATUS_KEYWORDS)
STRONG_DOMAIN_KEYWORDS = _expand_keyword_set_with_repair(STRONG_DOMAIN_KEYWORDS)
ML_GOAL_QUERY_KEYWORDS = _expand_keyword_set_with_repair(ML_GOAL_QUERY_KEYWORDS)
ML_SUCCESS_QUERY_KEYWORDS = _expand_keyword_set_with_repair(ML_SUCCESS_QUERY_KEYWORDS)
ML_GENERAL_PREDICTION_KEYWORDS = _expand_keyword_set_with_repair(ML_GENERAL_PREDICTION_KEYWORDS)

MOTIVATION_LINES = {
    "en": [
        "Your consistency lately is excellent.",
        "You are progressing step by step in the right direction.",
        "Even if progress feels slow, your discipline is working.",
        "What you are doing now will show clear results soon.",
        "Real progress starts with routine, and you are building it.",
        "You are doing better than you think.",
    ],
    "ar_fusha": [
        "\u0639\u0645\u0644\u0643 \u0645\u0645\u062a\u0627\u0632 \u0641\u064a \u0627\u0644\u0641\u062a\u0631\u0629 \u0627\u0644\u0623\u062e\u064a\u0631\u0629.",
        "\u0648\u0627\u0636\u062d \u0623\u0646\u0643 \u0645\u0644\u062a\u0632\u0645 \u0648\u062a\u062a\u0642\u062f\u0645 \u062e\u0637\u0648\u0629 \u0628\u062e\u0637\u0648\u0629.",
        "\u0623\u0646\u0627 \u0641\u062e\u0648\u0631 \u0628\u0627\u0644\u0627\u0644\u062a\u0632\u0627\u0645 \u0627\u0644\u0630\u064a \u062a\u0642\u062f\u0645\u0647.",
        "\u062d\u062a\u0649 \u0644\u0648 \u0643\u0627\u0646 \u0627\u0644\u062a\u0642\u062f\u0645 \u0628\u0637\u064a\u0626\u0627\u064b \u0641\u0623\u0646\u062a \u0639\u0644\u0649 \u0627\u0644\u0645\u0633\u0627\u0631 \u0627\u0644\u0635\u062d\u064a\u062d.",
        "\u0627\u0644\u0646\u062a\u0627\u0626\u062c \u0627\u0644\u062c\u064a\u062f\u0629 \u062a\u0628\u062f\u0623 \u0628\u0627\u0644\u0627\u0646\u0636\u0628\u0627\u0637.",
        "\u0627\u0633\u062a\u0645\u0631 \u2014 \u0623\u0646\u062a \u0645\u0627\u0634\u064d \u0628\u0634\u0643\u0644 \u0645\u0645\u062a\u0627\u0632.",
    ],
    "ar_jordanian": [
        "\u0634\u063a\u0644\u0643 \u0645\u0645\u062a\u0627\u0632 \u0628\u0627\u0644\u0641\u062a\u0631\u0629 \u0627\u0644\u0623\u062e\u064a\u0631\u0629.",
        "\u0648\u0627\u0636\u062d \u0625\u0646\u0643 \u0645\u0644\u062a\u0632\u0645 \u0648\u0639\u0645 \u062a\u062a\u0642\u062f\u0645 \u0634\u0648\u064a \u0634\u0648\u064a.",
        "\u062d\u062a\u0649 \u0644\u0648 \u0627\u0644\u062a\u0642\u062f\u0645 \u0628\u0637\u064a\u0621 \u2014 \u0625\u0646\u062a \u0645\u0627\u0634\u064a \u0635\u062d.",
        "\u0627\u0633\u062a\u0645\u0631\u060c \u0625\u0646\u062a \u0639\u0644\u0649 \u0627\u0644\u0645\u0633\u0627\u0631 \u0627\u0644\u0635\u062d.",
        "\u0627\u0644\u0646\u062a\u0627\u0626\u062c \u0628\u062f\u0647\u0627 \u0635\u0628\u0631 \u0628\u0633 \u0625\u0646\u062a \u0634\u063a\u0627\u0644 \u0635\u062d.",
        "\u0623\u0646\u0627 \u0645\u0639\u0643 \u062e\u0637\u0648\u0629 \u0628\u062e\u0637\u0648\u0629.",
    ],
}

def _normalize_user_id(user_id: Optional[str]) -> str:
    return (user_id or "anonymous").strip() or "anonymous"


def _normalize_conversation_id(conversation_id: Optional[str], user_id: str) -> str:
    return (conversation_id or f"conv_{user_id}").strip() or f"conv_{user_id}"


def _session_key(user_id: str, conversation_id: str) -> str:
    return f"{user_id}:{conversation_id}"


def _get_memory_session(user_id: str, conversation_id: str) -> MemorySystem:
    key = _session_key(user_id, conversation_id)
    if key not in MEMORY_SESSIONS:
        MEMORY_SESSIONS[key] = MemorySystem(user_id=user_id, max_short_term=10)
    return MEMORY_SESSIONS[key]


def _get_user_state(user_id: str) -> Dict[str, Any]:
    if user_id not in USER_STATE:
        USER_STATE[user_id] = {}
    return USER_STATE[user_id]


def _contains_any(text: str, keywords: set[str]) -> bool:
    return fuzzy_contains_any(text, keywords)


def _contains_phrase(text: str, phrases: set[str]) -> bool:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return False
    for phrase in phrases:
        phrase_norm = normalize_text(phrase)
        if phrase_norm and phrase_norm in normalized_text:
            return True
    return False


def _is_nutrition_knowledge_query(user_input: str) -> bool:
    normalized = normalize_text(user_input)
    if not normalized:
        return False
    return _contains_any(normalized, NUTRITION_KB_KEYWORDS | NUTRITION_PLAN_KEYWORDS)


def _is_greeting_query(user_input: str) -> bool:
    normalized = normalize_text(user_input)
    if not normalized:
        return False
    if _dataset_intent_matches(user_input, "greeting"):
        return True
    if len(normalized.split()) > 4:
        return False
    greeting_phrases = {
        "hi",
        "hello",
        "hey",
        "مرحبا",
        "اهلا",
        "هلا",
        "السلام عليكم",
        "سلام",
    }
    return _contains_phrase(normalized, greeting_phrases)


def _is_name_query(user_input: str) -> bool:
    return _contains_phrase(
        user_input,
        {
            "what is your name",
            "your name",
            "name",
            "اسمك",
            "شو اسمك",
            "مين انت",
            "من انت",
        },
    )


def _is_how_are_you_query(user_input: str) -> bool:
    return _contains_phrase(
        user_input,
        {
            "how are you",
            "كيفك",
            "كيف حالك",
            "شلونك",
            "كيف الحال",
        },
    )


def _is_workout_plan_request(user_input: str) -> bool:
    normalized = normalize_text(user_input)
    plan_terms = {
        "plan",
        "program",
        "schedule",
        "weekly",
        "خطة",
        "جدول",
        "برنامج",
        "اسبوع",
        "أسبوع",
    }
    workout_terms = {
        "workout",
        "training",
        "exercise",
        "gym",
        "تمرين",
        "تمارين",
        "تدريب",
        "عضل",
        "عضلات",
    }
    return _contains_any(normalized, plan_terms) and _contains_any(normalized, workout_terms)


def _is_nutrition_plan_request(user_input: str) -> bool:
    normalized = normalize_text(user_input)
    plan_terms = {
        "plan",
        "program",
        "schedule",
        "daily",
        "خطة",
        "جدول",
        "برنامج",
        "يومي",
        "يومية",
    }
    nutrition_terms = {
        "nutrition",
        "diet",
        "meal",
        "calories",
        "food",
        "تغذية",
        "وجبات",
        "اكل",
        "طعام",
        "سعرات",
    }
    return _contains_any(normalized, plan_terms) and _contains_any(normalized, nutrition_terms)


def _is_generic_plan_request(user_input: str) -> bool:
    normalized = normalize_text(user_input)
    if not normalized:
        return False

    plan_terms = {
        "plan",
        "program",
        "schedule",
        "routine",
        "خطة",
        "برنامج",
        "جدول",
        "بلان",
    }
    if not _contains_any(normalized, plan_terms):
        return False

    # Not generic if already explicit.
    if _is_workout_plan_request(user_input) or _is_nutrition_plan_request(user_input):
        return False
    return True


def _resolve_plan_type_from_message(user_input: str) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    if _is_workout_plan_request(user_input):
        return "workout", None
    if _is_nutrition_plan_request(user_input):
        return "nutrition", None
    if not _is_generic_plan_request(user_input):
        return None, None

    try:
        prediction = predict_plan_intent(user_input)
        predicted = str(prediction.get("predicted_intent", "")).strip().lower()
        confidence = _to_float(prediction.get("confidence"))
        if predicted in {"workout", "nutrition"} and (confidence is None or confidence >= 0.50):
            return predicted, prediction
    except FileNotFoundError:
        return None, None
    except Exception:
        return None, None

    return None, None


def _infer_goal_for_plan(profile: dict[str, Any], tracking_summary: Optional[dict[str, Any]]) -> tuple[str, Optional[float], bool]:
    explicit = _normalize_goal(profile.get("goal"))
    if explicit in {"muscle_gain", "fat_loss", "general_fitness"}:
        return explicit, None, False

    payload, _missing = _build_goal_prediction_payload(profile, tracking_summary)
    try:
        prediction = predict_goal(payload)
    except Exception:
        return "general_fitness", None, True

    predicted = _normalize_goal(prediction.get("predicted_goal"))
    confidence = None
    probs = prediction.get("probabilities") if isinstance(prediction.get("probabilities"), dict) else {}
    if predicted in probs:
        confidence = _to_float(probs.get(predicted))

    if predicted not in {"muscle_gain", "fat_loss", "general_fitness"}:
        predicted = "general_fitness"
    return predicted, confidence, True


def _has_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


def _detect_language(requested_language: str, message: str, profile: dict[str, Any]) -> str:
    requested = (requested_language or "en").strip().lower()
    repaired_message = _repair_mojibake(message or "")

    # Always prioritize the actual message content so Arabic works even if UI language is English.
    if _has_arabic(repaired_message):
        preferred = str(profile.get("preferred_language", "")).lower()
        if preferred in {"ar_fusha", "ar_jordanian"}:
            return preferred

        lowered = normalize_text(repaired_message)
        if any(token in lowered for token in JORDANIAN_HINTS):
            return "ar_jordanian"
        return "ar_fusha"

    if requested in {"ar_fusha", "ar_jordanian"}:
        return requested

    if requested == "ar":
        preferred = str(profile.get("preferred_language", "")).lower()
        if preferred in {"ar_fusha", "ar_jordanian"}:
            return preferred
        return "ar_fusha"

    return "en"


def _parse_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        split_tokens = re.split(r"[,،\n]| and | و ", _repair_mojibake(value))
        return [t.strip() for t in split_tokens if t.strip()]
    return [str(value).strip()]


def _nutrition_kb_context(user_input: str, profile: dict[str, Any], top_k: int = 3) -> str:
    if not NUTRITION_KB.ready:
        return ""
    if not _is_nutrition_knowledge_query(user_input):
        return ""

    query_parts: list[str] = [user_input]
    goal = str(profile.get("goal", "")).strip()
    if goal:
        query_parts.append(goal)

    chronic_diseases = _parse_list_field(profile.get("chronic_diseases"))
    allergies = _parse_list_field(profile.get("allergies"))
    if chronic_diseases:
        query_parts.append(" ".join(chronic_diseases))
    if allergies:
        query_parts.append(" ".join(allergies))

    query = " | ".join(part for part in query_parts if part)
    hits = NUTRITION_KB.search(query, top_k=top_k, max_chars=420)
    if not hits:
        return ""
    return "\n".join(f"- {hit['text']}" for hit in hits)


def _normalize_goal(goal: Any) -> str:
    text = normalize_text(str(goal or ""))
    if not text:
        return ""
    if fuzzy_contains_any(
        text,
        {
            "bulking",
            "muscle gain",
            "gain muscle",
            "build muscle",
            "hypertrophy",
            "تضخيم",
            "زيادة عضل",
            "بناء عضل",
        },
    ):
        return "muscle_gain"
    if fuzzy_contains_any(
        text,
        {
            "cutting",
            "fat loss",
            "lose fat",
            "lose weight",
            "weight loss",
            "تنشيف",
            "خسارة وزن",
            "نزول وزن",
            "حرق دهون",
        },
    ):
        return "fat_loss"
    if fuzzy_contains_any(text, {"fitness", "general fitness", "health", "maintenance", "لياقة", "رشاقة", "صحة"}):
        return "general_fitness"
    if text in {"bulking", "muscle_gain", "gain muscle", "build muscle", "زيادة عضل", "بناء عضل"}:
        return "muscle_gain"
    if text in {"cutting", "fat_loss", "lose fat", "lose weight", "تنشيف", "خسارة وزن"}:
        return "fat_loss"
    if text in {"fitness", "general_fitness", "لياقة", "رشاقة"}:
        return "general_fitness"
    return text


def _dataset_text(value: Any, language: str = "en") -> str:
    if isinstance(value, dict):
        en_text = _repair_mojibake(str(value.get("en", "")).strip())
        ar_text = _repair_mojibake(str(value.get("ar", "")).strip())
        if language == "en":
            return en_text or ar_text
        return ar_text or en_text
    return _repair_mojibake(str(value or "").strip())


def _dataset_goal_key(value: Any) -> str:
    if isinstance(value, dict):
        text = f"{value.get('en', '')} {value.get('ar', '')}".strip()
    else:
        text = str(value or "")
    return _normalize_goal(text)


def _dataset_level_key(value: Any) -> str:
    normalized = normalize_text(str(value or ""))
    if "beg" in normalized or "مبت" in normalized:
        return "beginner"
    if "inter" in normalized or "متوس" in normalized:
        return "intermediate"
    if "adv" in normalized or "متقد" in normalized:
        return "advanced"
    return "beginner"


def _dataset_intent_matches(user_input: str, tag: str) -> bool:
    return RESPONSE_DATASETS.matches_intent(user_input, tag)


def _dataset_intent_response(tag: str, language: str, seed: str = "") -> Optional[str]:
    response = RESPONSE_DATASETS.pick_response(tag, language=language, seed=seed)
    if not response:
        return None
    return _repair_mojibake(response)


def _dataset_conversation_reply(user_input: str, language: str) -> Optional[str]:
    # Priority order for conversational intents loaded from the provided dataset.
    ordered_tags: list[str] = [
        "greeting",
        "gratitude",
        "goodbye",
        "ask_exercise",
        "ask_muscle",
        "ask_home_workout",
        "ask_gym_workout",
        "ask_weight_loss",
        "ask_muscle_gain",
        "ask_general_fitness",
    ]
    known_tags = set(RESPONSE_DATASETS.intents.keys())
    for tag in ordered_tags:
        if tag not in known_tags:
            continue
        if _dataset_intent_matches(user_input, tag):
            return _dataset_intent_response(tag, language, seed=user_input)

    # Include any additional tags from dataset except fallback/sample buckets.
    for tag in RESPONSE_DATASETS.intents.keys():
        if tag in set(ordered_tags) or tag in {"out_of_scope", "short_conversations"}:
            continue
        if _dataset_intent_matches(user_input, tag):
            return _dataset_intent_response(tag, language, seed=user_input)
    return None


def _dataset_fallback_reply(language: str, seed: str = "") -> str:
    for tag in ("out_of_scope", "greeting", "gratitude", "goodbye"):
        response = _dataset_intent_response(tag, language, seed=seed)
        if response:
            return response
    return "Unable to respond."


def _strict_out_of_scope_reply(language: str) -> str:
    return _lang_reply(
        language,
        "This assistant is specialized only in fitness topics: workouts, nutrition, body composition, recovery, and progress tracking.",
        "هذا المساعد متخصص فقط في مواضيع اللياقة: التمارين، التغذية، تركيب الجسم، التعافي، ومتابعة التقدم.",
        "هذا المساعد متخصص بس بمواضيع اللياقة: التمارين، التغذية، تركيب الجسم، التعافي، ومتابعة التقدم.",
    )


def _generate_workout_plan_options_from_dataset(
    profile: dict[str, Any],
    language: str,
    count: int = 5,
) -> list[dict[str, Any]]:
    programs = RESPONSE_DATASETS.workout_programs
    if not isinstance(programs, list) or not programs:
        return []

    goal_key = _normalize_goal(profile.get("goal") or "general_fitness")
    level_key = str(profile.get("fitness_level", "beginner")).lower()
    if level_key not in {"beginner", "intermediate", "advanced"}:
        level_key = "beginner"

    scored_programs: list[tuple[int, dict[str, Any]]] = []
    for program in programs:
        if not isinstance(program, dict):
            continue
        score = 0
        program_goal = _dataset_goal_key(program.get("goal"))
        if program_goal == goal_key:
            score += 2
        program_level = _dataset_level_key(program.get("level"))
        if program_level == level_key:
            score += 1
        scored_programs.append((score, program))

    scored_programs.sort(key=lambda item: item[0], reverse=True)
    selected = [item[1] for item in scored_programs[: max(1, min(count, len(scored_programs)))]]

    rest_days = [d for d in profile.get("rest_days", []) if isinstance(d, str) and any(d == wd[0] for wd in WEEK_DAYS)]
    options: list[dict[str, Any]] = []

    for program in selected:
        program_days = [d for d in program.get("days", []) if isinstance(d, dict)]
        if not program_days:
            continue
        program_days = sorted(program_days, key=lambda d: int(d.get("day_number", 0) or 0))

        days_per_week = int(program.get("days_per_week", len(program_days)) or len(program_days) or 3)
        days_per_week = max(1, min(7, days_per_week))

        if not rest_days:
            rest_count = max(0, 7 - days_per_week)
            rest_days_local = [day for day, _ in WEEK_DAYS[-rest_count:]] if rest_count else []
        else:
            rest_days_local = rest_days[:]

        training_days = [day for day, _ in WEEK_DAYS if day not in rest_days_local][:days_per_week]
        if len(training_days) < days_per_week:
            for day, _ in WEEK_DAYS:
                if day not in training_days:
                    training_days.append(day)
                if len(training_days) >= days_per_week:
                    break

        training_day_payload: dict[str, dict[str, Any]] = {}
        for idx, day_name in enumerate(training_days):
            source_day = program_days[idx % len(program_days)]
            exercises_raw = source_day.get("exercises", [])
            exercises: list[dict[str, Any]] = []
            for ex in exercises_raw:
                if not isinstance(ex, dict):
                    continue
                name_en = _dataset_text(ex.get("name"), "en") or "Exercise"
                name_ar = _dataset_text(ex.get("name"), "ar_fusha") or name_en
                reps = str(ex.get("reps", "8-12"))
                sets = str(ex.get("sets", 3))
                rest_seconds = int(_to_float(ex.get("rest_seconds")) or 60)
                exercises.append(
                    {
                        "name": name_en,
                        "nameAr": name_ar,
                        "sets": sets,
                        "reps": reps,
                        "rest_seconds": rest_seconds,
                        "notes": "",
                    }
                )

            training_day_payload[day_name] = {
                "focus": _dataset_text(source_day.get("focus"), language) or "Workout",
                "exercises": exercises,
            }

        normalized_days: list[dict[str, Any]] = []
        for day_en, day_ar in WEEK_DAYS:
            payload = training_day_payload.get(day_en)
            if payload:
                normalized_days.append(
                    {
                        "day": day_en,
                        "dayAr": day_ar,
                        "focus": payload.get("focus", "Workout"),
                        "exercises": payload.get("exercises", []),
                    }
                )
            else:
                normalized_days.append({"day": day_en, "dayAr": day_ar, "focus": "Rest", "exercises": []})

        title_en = _dataset_text(program.get("name"), "en") or "Workout Plan"
        title_ar = _dataset_text(program.get("name"), "ar_fusha") or title_en
        goal = _dataset_goal_key(program.get("goal")) or goal_key

        options.append(
            {
                "id": f"workout_{uuid.uuid4().hex[:10]}",
                "type": "workout",
                "title": title_en,
                "title_ar": title_ar,
                "goal": goal,
                "fitness_level": _dataset_level_key(program.get("level")),
                "rest_days": [d["day"] for d in normalized_days if not d.get("exercises")],
                "duration_days": 7,
                "days": normalized_days,
                "created_at": datetime.utcnow().isoformat(),
                "source": "week2_workout_programs_dataset",
            }
        )

    return options


def _generate_nutrition_plan_options_from_dataset(
    profile: dict[str, Any],
    language: str,
    count: int = 5,
) -> list[dict[str, Any]]:
    programs = RESPONSE_DATASETS.nutrition_programs
    if not isinstance(programs, list) or not programs:
        return []

    goal_key = _normalize_goal(profile.get("goal") or "general_fitness")
    current_weight = _to_float(profile.get("weight"))

    scored_programs: list[tuple[int, dict[str, Any]]] = []
    for program in programs:
        if not isinstance(program, dict):
            continue
        score = 0
        program_goal = _dataset_goal_key(program.get("goal"))
        if program_goal == goal_key:
            score += 2
        range_payload = program.get("weight_range_kg", {}) if isinstance(program.get("weight_range_kg"), dict) else {}
        min_w = _to_float(range_payload.get("min"))
        max_w = _to_float(range_payload.get("max"))
        if current_weight is not None and min_w is not None and max_w is not None and min_w <= current_weight <= max_w:
            score += 1
        scored_programs.append((score, program))

    scored_programs.sort(key=lambda item: item[0], reverse=True)
    selected = [item[1] for item in scored_programs[: max(1, min(count, len(scored_programs)))]]

    options: list[dict[str, Any]] = []
    for program in selected:
        restrictions = _build_food_restrictions(profile)
        calorie_range = program.get("calorie_range", {}) if isinstance(program.get("calorie_range"), dict) else {}
        cal_min = int(_to_float(calorie_range.get("min")) or 1800)
        cal_max = int(_to_float(calorie_range.get("max")) or max(cal_min, 2000))
        daily_calories = int(round((cal_min + cal_max) / 2))

        macro = program.get("macro_split", {}) if isinstance(program.get("macro_split"), dict) else {}
        protein_pct = _to_float(macro.get("protein_pct")) or 30.0
        carbs_pct = _to_float(macro.get("carbs_pct")) or 45.0
        fat_pct = _to_float(macro.get("fat_pct")) or 25.0

        sample_meals = [m for m in program.get("sample_meals", []) if isinstance(m, dict)]
        if not sample_meals:
            sample_meals = [{"meal_type": "Meal", "description": "Balanced meal"}]
        sample_meals = _filter_meals_by_restrictions(sample_meals, restrictions.get("tokens", set()))

        meals_per_day = int(profile.get("meals_per_day") or len(sample_meals) or 3)
        meals_per_day = max(2, min(6, meals_per_day))
        calories_per_meal = max(120, int(round(daily_calories / meals_per_day)))

        days_payload: list[dict[str, Any]] = []
        for day_en, day_ar in WEEK_DAYS:
            meals: list[dict[str, Any]] = []
            for i in range(meals_per_day):
                template = sample_meals[i % len(sample_meals)]
                meal_name_en = _dataset_text(template.get("meal_type"), "en") or f"Meal {i + 1}"
                meal_name_ar = _dataset_text(template.get("meal_type"), "ar_fusha") or meal_name_en
                meal_desc_en = _dataset_text(template.get("description"), "en")
                meal_desc_ar = _dataset_text(template.get("description"), "ar_fusha") or meal_desc_en
                meals.append(
                    {
                        "name": meal_name_en,
                        "nameAr": meal_name_ar,
                        "description": meal_desc_en,
                        "descriptionAr": meal_desc_ar,
                        "calories": str(calories_per_meal),
                        "time": f"meal_{i + 1}",
                    }
                )
            days_payload.append({"day": day_en, "dayAr": day_ar, "meals": meals})

        title_goal_en = _dataset_text(program.get("goal"), "en") or "Nutrition Plan"
        title_goal_ar = _dataset_text(program.get("goal"), "ar_fusha") or title_goal_en
        tips = program.get("tips", []) if isinstance(program.get("tips"), list) else []
        tips_text = " ".join(_dataset_text(tip, language) for tip in tips if str(tip).strip())
        if restrictions.get("labels"):
            tips_text = " ".join([tips_text, f"Avoid: {', '.join(restrictions['labels'])}."]).strip()
        est_protein = int(round((daily_calories * (protein_pct / 100.0)) / 4.0))

        options.append(
            {
                "id": f"nutrition_{uuid.uuid4().hex[:10]}",
                "type": "nutrition",
                "title": f"{title_goal_en} - Nutrition Plan",
                "title_ar": f"{title_goal_ar} - خطة تغذية",
                "goal": _dataset_goal_key(program.get("goal")) or goal_key,
                "daily_calories": daily_calories,
                "estimated_protein": est_protein,
                "meals_per_day": meals_per_day,
                "days": days_payload,
                "notes": tips_text,
                "macro_split": {"protein_pct": protein_pct, "carbs_pct": carbs_pct, "fat_pct": fat_pct},
                "forbidden_foods": list(restrictions.get("labels", [])),
                "created_at": datetime.utcnow().isoformat(),
                "source": "week2_nutrition_programs_dataset",
            }
        )

    return options


def _build_profile(req: ChatRequest, user_state: dict[str, Any]) -> dict[str, Any]:
    profile = dict(req.user_profile or {})

    if "chronicConditions" in profile and "chronic_diseases" not in profile:
        profile["chronic_diseases"] = _parse_list_field(profile.get("chronicConditions"))
    if "allergies" in profile:
        profile["allergies"] = _parse_list_field(profile.get("allergies"))
    if "chronic_diseases" in profile:
        profile["chronic_diseases"] = _parse_list_field(profile.get("chronic_diseases"))

    profile["goal"] = _normalize_goal(profile.get("goal"))

    for key in (
        "goal",
        "fitness_level",
        "rest_days",
        "age",
        "weight",
        "height",
        "gender",
        "meals_per_day",
        "allergies",
        "chronic_diseases",
        "target_calories",
        "preferred_language",
    ):
        if key in user_state and user_state[key] is not None:
            profile[key] = user_state[key]

    return profile


def _lang_reply(language: str, en: str, ar_fusha: str, ar_jordanian: Optional[str] = None) -> str:
    if language == "en":
        return _repair_mojibake(en)
    if language == "ar_fusha":
        return _repair_mojibake(ar_fusha)
    return _repair_mojibake(ar_jordanian or ar_fusha)


def _motivation_line(language: str, seed: str = "") -> str:
    lines = MOTIVATION_LINES.get(language) or MOTIVATION_LINES["en"]
    if not lines:
        return ""
    idx = abs(hash(seed or "default")) % len(lines)
    return lines[idx]


def _persist_profile_context(profile: dict[str, Any], state: dict[str, Any]) -> None:
    tracked_keys = (
        "name",
        "goal",
        "fitness_level",
        "rest_days",
        "age",
        "weight",
        "height",
        "gender",
        "meals_per_day",
        "allergies",
        "chronic_diseases",
        "target_calories",
        "preferred_language",
    )
    for key in tracked_keys:
        value = profile.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, list) and not value:
            continue
        state[key] = value


def _profile_display_name(profile: dict[str, Any]) -> str:
    for key in ("name", "full_name", "first_name"):
        value = profile.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _profile_goal_label(goal: str, language: str) -> str:
    goal_key = str(goal or "").strip().lower()
    if goal_key == "muscle_gain":
        return _lang_reply(language, "muscle gain", "زيادة الكتلة العضلية", "زيادة العضل")
    if goal_key == "fat_loss":
        return _lang_reply(language, "fat loss", "خسارة الدهون", "تنزيل الدهون")
    if goal_key == "general_fitness":
        return _lang_reply(language, "general fitness", "اللياقة العامة", "لياقة عامة")
    return str(goal or "")


def _profile_query_reply(
    user_input: str,
    language: str,
    profile: dict[str, Any],
    tracking_summary: Optional[dict[str, Any]],
) -> Optional[str]:
    name = _profile_display_name(profile)
    goal_label = _profile_goal_label(str(profile.get("goal", "")), language)
    age = profile.get("age")
    height = profile.get("height")
    weight = profile.get("weight")
    normalized = normalize_text(user_input)

    if _contains_any(normalized, WHO_AM_I_KEYWORDS):
        if name:
            return _lang_reply(
                language,
                f"You are {name}. I have your profile and can coach you using your goal, body stats, and progress.",
                f"أنت {name}. لدي ملفك الشخصي، وأستطيع تدريبك وفق هدفك وقياساتك وتقدمك.",
                f"إنت {name}. عندي ملفك، وبقدر أدربك حسب هدفك وقياساتك وتقدمك.",
            )
        return _lang_reply(
            language,
            "I do not have your name yet. Add it in your profile page and I will personalize every response.",
            "لا أملك اسمك بعد. أضفه في صفحة الملف الشخصي وسأخصص كل الردود لك.",
            "لسا ما عندي اسمك. حطه بصفحة البروفايل وبخصصلك كل الردود.",
        )

    if _contains_any(normalized, ASK_MY_AGE_KEYWORDS):
        if age is not None:
            return _lang_reply(
                language,
                f"Your age is {age}.",
                f"عمرك هو {age}.",
                f"عمرك {age}.",
            )
        return _lang_reply(
            language,
            "I do not have your age yet. Update it in your profile and I will use it in your plans.",
            "لا أملك عمرك بعد. حدّثه في الملف الشخصي وسأستخدمه في خططك.",
            "لسا ما عندي عمرك. حدّثه بالبروفايل وبستخدمه بخططك.",
        )

    if _contains_any(normalized, ASK_MY_HEIGHT_KEYWORDS):
        if height is not None:
            return _lang_reply(
                language,
                f"Your height is {height} cm.",
                f"طولك هو {height} سم.",
                f"طولك {height} سم.",
            )
        return _lang_reply(
            language,
            "I do not have your height yet. Add it in your profile to make training and calories more accurate.",
            "لا أملك طولك بعد. أضفه في ملفك لتحسين دقة التدريب والسعرات.",
            "لسا ما عندي طولك. أضفه بالبروفايل عشان أدق بالتمارين والسعرات.",
        )

    if _contains_any(normalized, ASK_MY_WEIGHT_KEYWORDS):
        if weight is not None:
            return _lang_reply(
                language,
                f"Your weight is {weight} kg.",
                f"وزنك هو {weight} كغ.",
                f"وزنك {weight} كيلو.",
            )
        return _lang_reply(
            language,
            "I do not have your weight yet. Add it in your profile and I will tune your plan calories better.",
            "لا أملك وزنك بعد. أضفه في ملفك وسأضبط سعرات الخطة بدقة أعلى.",
            "لسا ما عندي وزنك. أضفه بالبروفايل وبضبطلك السعرات أدق.",
        )

    if _contains_any(normalized, ASK_MY_GOAL_KEYWORDS):
        if goal_label:
            return _lang_reply(
                language,
                f"Your current goal is {goal_label}.",
                f"هدفك الحالي هو: {goal_label}.",
                f"هدفك الحالي: {goal_label}.",
            )
        return _lang_reply(
            language,
            "Your goal is not set yet. Tell me if you want muscle gain, fat loss, or general fitness.",
            "هدفك غير محدد بعد. أخبرني: زيادة عضل أم خسارة دهون أم لياقة عامة.",
            "لسا هدفك مش محدد. احكيلي: زيادة عضل ولا تنزيل دهون ولا لياقة عامة.",
        )

    if _contains_any(normalized, {"my progress summary", "ملخص تقدمي", "ملخص التقدم"}):
        return _tracking_reply(language, tracking_summary)

    return None


def _social_reply(user_input: str, language: str, profile: dict[str, Any]) -> Optional[str]:
    normalized = normalize_text(user_input)
    name = _profile_display_name(profile)
    name_suffix = f" {name}" if name else ""

    if _dataset_intent_matches(user_input, "gratitude") or _contains_any(normalized, THANKS_KEYWORDS):
        dataset_reply = _dataset_intent_response("gratitude", language, seed=name or user_input)
        if dataset_reply:
            return dataset_reply
        return _lang_reply(
            language,
            f"Anytime{name_suffix}. Keep going and send me your next update.",
            f"على الرحب والسعة{name_suffix}. استمر وأرسل لي تحديثك التالي.",
            f"على راسي{name_suffix}. كمل وابعثلي تحديثك الجاي.",
        )

    if _dataset_intent_matches(user_input, "goodbye"):
        dataset_reply = _dataset_intent_response("goodbye", language, seed=name or user_input)
        if dataset_reply:
            return dataset_reply

    return None


def _plan_status_reply(language: str, plan_snapshot: Optional[dict[str, Any]]) -> str:
    if not plan_snapshot:
        return _lang_reply(
            language,
            "I do not have your latest plan status yet. Open your Schedule page and I can sync after your next message.",
            "Ù„Ø§ Ø£Ù…Ù„Ùƒ Ø¢Ø®Ø± Ø­Ø§Ù„Ø© Ù„Ø®Ø·Ø·Ùƒ Ø¨Ø¹Ø¯. Ø§ÙØªØ­ ØµÙØ­Ø© Ø§Ù„Ø¬Ø¯ÙˆÙ„ ÙˆØ³Ø£Ø²Ø§Ù…Ù†Ù‡Ø§ Ø¨Ø¹Ø¯ Ø±Ø³Ø§Ù„ØªÙƒ Ø§Ù„ØªØ§Ù„ÙŠØ©.",
            "Ù„Ø³Ø§ Ù…Ø§ Ø¹Ù†Ø¯ÙŠ Ø¢Ø®Ø± Ø­Ø§Ù„Ø© Ù„Ù„Ø®Ø·Ø·. Ø§ÙØªØ­ ØµÙØ­Ø© Ø§Ù„Ø¬Ø¯ÙˆÙ„ ÙˆØ¨Ø±Ø¬Ø¹ Ø¨Ø²Ø§Ù…Ù†Ù‡Ø§ Ù…Ø¹Ùƒ Ø¨Ø¹Ø¯ Ø±Ø³Ø§Ù„ØªÙƒ Ø§Ù„Ø¬Ø§ÙŠØ©.",
        )

    workout_count = int(plan_snapshot.get("active_workout_plans", 0) or 0)
    nutrition_count = int(plan_snapshot.get("active_nutrition_plans", 0) or 0)
    return _lang_reply(
        language,
        f"You currently have {workout_count} active workout plan(s) and {nutrition_count} active nutrition plan(s).",
        f"Ù„Ø¯ÙŠÙƒ Ø­Ø§Ù„ÙŠÙ‹Ø§ {workout_count} Ø®Ø·Ø© ØªÙ…Ø§Ø±ÙŠÙ† Ù†Ø´Ø·Ø© Ùˆ{nutrition_count} Ø®Ø·Ø© ØªØºØ°ÙŠØ© Ù†Ø´Ø·Ø©.",
        f"Ø­Ø§Ù„ÙŠÙ‹Ø§ Ø¹Ù†Ø¯Ùƒ {workout_count} Ø®Ø·Ø© ØªÙ…Ø§Ø±ÙŠÙ† ÙØ¹Ø§Ù„Ø© Ùˆ{nutrition_count} Ø®Ø·Ø© ØªØºØ°ÙŠØ© ÙØ¹Ø§Ù„Ø©.",
    )


def _progress_diagnostic_reply(language: str, profile: dict[str, Any], tracking_summary: Optional[dict[str, Any]]) -> str:
    adherence = 0.0
    if tracking_summary:
        try:
            adherence = float(tracking_summary.get("adherence_score", 0) or 0)
        except (TypeError, ValueError):
            adherence = 0.0
    adherence_pct = int(round(adherence * 100))
    weight = profile.get("weight")
    try:
        hydration_liters = round(max(1.8, float(weight) * 0.033), 1) if weight is not None else 2.5
    except (TypeError, ValueError):
        hydration_liters = 2.5

    return _lang_reply(
        language,
        (
            f"Plateaus are common. Your adherence is about {adherence_pct}%. "
            "Let us find the cause step by step:\n"
            "1. How many hours do you sleep on average?\n"
            f"2. Do you drink around {hydration_liters}L water daily?\n"
            "3. Are you completing your planned sets/reps, or stopping early?\n"
            "4. Are you consistently hitting your calories and protein targets?\n"
            "Reply with these 4 points and I will give you a precise fix."
        ),
        (
            f"ثبات النتائج أمر طبيعي أحيانًا. نسبة التزامك الحالية تقريبًا {adherence_pct}%.\n"
            "لنحدد السبب خطوة بخطوة:\n"
            "1. كم ساعة تنام يوميًا بالمتوسط؟\n"
            f"2. هل تشرب تقريبًا {hydration_liters} لتر ماء يوميًا؟\n"
            "3. هل تكمل المجموعات والتكرارات كاملة أم تتوقف مبكرًا؟\n"
            "4. هل تلتزم يوميًا بسعراتك وبروتينك المستهدف؟\n"
            "أجبني على هذه النقاط الأربع وسأعطيك الحل الأدق."
        ),
        (
            f"ثبات الجسم بصير، والتزامك الحالي تقريبًا {adherence_pct}%.\n"
            "خلينا نعرف السبب شوي شوي:\n"
            "1. كم ساعة نومك بالمتوسط؟\n"
            f"2. بتشرب تقريبًا {hydration_liters} لتر مي باليوم؟\n"
            "3. بتكمل كل المجموعات والتكرارات ولا بتوقف بكير؟\n"
            "4. ملتزم بسعراتك وبروتينك يوميًا؟\n"
            "جاوبني بهدول الأربع نقاط وبعطيك الحل الأدق."
        ),
    )


def _exercise_diagnostic_reply(language: str) -> str:
    return _lang_reply(
        language,
        (
            "Understood. To fix your exercise form safely, answer these points:\n"
            "1. Which exercise exactly?\n"
            "2. Where do you feel pain/tension?\n"
            "3. At which rep does form break down?\n"
            "4. What load are you using now?\n"
            "5. Did this start after an injury or sudden volume increase?\n"
            "After your answers, I will give exact technique corrections and load changes."
        ),
        (
            "ممتاز، لنصحح أداء التمرين بشكل آمن أجبني على التالي:\n"
            "1. ما اسم التمرين بالضبط؟\n"
            "2. أين تشعر بالألم أو الشد؟\n"
            "3. في أي تكرار يبدأ الأداء بالانهيار؟\n"
            "4. ما الوزن الذي تستخدمه الآن؟\n"
            "5. هل بدأ هذا بعد إصابة أو زيادة مفاجئة في الحمل التدريبي؟\n"
            "بعد إجاباتك أعطيك تصحيحًا دقيقًا للحركة وتعديلًا مناسبًا للأوزان."
        ),
        (
            "تمام، عشان نصلح الأداء بدون إصابة جاوبني:\n"
            "1. شو اسم التمرين بالزبط؟\n"
            "2. وين بتحس بالألم أو الشد؟\n"
            "3. بأي تكرار بتخرب الحركة؟\n"
            "4. كم الوزن اللي بتلعب فيه هسا؟\n"
            "5. المشكلة بلشت بعد إصابة أو زيادة حمل مفاجئة؟\n"
            "بعدها بعطيك تصحيح دقيق للحركة وتعديل الوزن."
        ),
    )


def _normalize_recent_messages(raw_messages: Optional[list[dict[str, Any]]]) -> list[dict[str, str]]:
    if not raw_messages:
        return []
    cleaned: list[dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = _repair_mojibake(str(item.get("content", "")).strip())
        if not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned[-12:]


def _update_plan_snapshot_state(state: dict[str, Any], new_snapshot: Optional[dict[str, Any]]) -> None:
    if not new_snapshot:
        return

    previous = state.get("plan_snapshot")
    state["plan_snapshot"] = new_snapshot

    if not isinstance(previous, dict):
        return

    previous_total = int(previous.get("active_workout_plans", 0) or 0) + int(previous.get("active_nutrition_plans", 0) or 0)
    new_total = int(new_snapshot.get("active_workout_plans", 0) or 0) + int(new_snapshot.get("active_nutrition_plans", 0) or 0)

    if new_total < previous_total:
        state["plans_recently_deleted"] = True
    elif new_total >= previous_total:
        state["plans_recently_deleted"] = False


def _missing_fields_for_plan(plan_type: str, profile: dict[str, Any]) -> list[str]:
    if plan_type == "workout":
        required = ["goal", "fitness_level", "rest_days"]
    else:
        required = ["goal", "age", "weight", "height", "gender", "meals_per_day", "chronic_diseases", "allergies"]

    missing: list[str] = []
    for key in required:
        value = profile.get(key)
        if value is None:
            missing.append(key)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(key)
            continue
        if key == "rest_days" and (not isinstance(value, list) or len(value) == 0):
            missing.append(key)
    return missing


def _missing_field_question(field_name: str, language: str) -> str:
    questions = {
        "en": {
            "goal": "What is your main goal now: muscle gain, fat loss, or general fitness?",
            "fitness_level": "What is your current fitness level: beginner, intermediate, or advanced?",
            "rest_days": "Which days do you want as rest days this week?",
            "age": "What is your age?",
            "weight": "What is your current weight in kg?",
            "height": "What is your height in cm?",
            "gender": "What is your gender (male/female)?",
            "meals_per_day": "How many meals do you want per day (3, 4, or 5)?",
            "chronic_diseases": "Do you have any chronic diseases I should consider? If none, reply with 'none'.",
            "allergies": "Do you have any food allergies? If none, reply with 'none'.",
        },
        "ar_fusha": {
            "goal": "ما هو هدفك الرئيسي الآن: بناء عضل أم خسارة دهون أم لياقة عامة؟",
            "fitness_level": "ما هو مستواك الرياضي الحالي: مبتدئ أم متوسط أم متقدم؟",
            "rest_days": "ما هي أيام الراحة التي تريدها هذا الأسبوع؟",
            "age": "كم عمرك؟",
            "weight": "ما وزنك الحالي بالكيلوغرام؟",
            "height": "ما طولك بالسنتيمتر؟",
            "gender": "ما جنسك (ذكر/أنثى)؟",
            "meals_per_day": "كم وجبة تريد يوميًا (3 أو 4 أو 5)؟",
            "chronic_diseases": "هل لديك أمراض مزمنة يجب أخذها بالحسبان؟ إذا لا يوجد اكتب: لا يوجد",
            "allergies": "هل لديك أي حساسية غذائية؟ إذا لا يوجد اكتب: لا يوجد",
        },
        "ar_jordanian": {
            "goal": "شو هدفك هلأ: زيادة عضل، نزول دهون، ولا لياقة عامة؟",
            "fitness_level": "شو مستواك الرياضي: مبتدئ، متوسط، ولا متقدم؟",
            "rest_days": "أي أيام بدك تكون أيام راحة بالأسبوع؟",
            "age": "كم عمرك؟",
            "weight": "شو وزنك الحالي بالكيلو؟",
            "height": "كم طولك بالسنتي؟",
            "gender": "شو جنسك (ذكر/أنثى)؟",
            "meals_per_day": "كم وجبة بدك باليوم (3 أو 4 أو 5)؟",
            "chronic_diseases": "في أمراض مزمنة لازم آخدها بالحسبان؟ إذا ما في اكتب: ما في",
            "allergies": "عندك حساسية أكل؟ إذا ما في اكتب: ما في",
        },
    }
    return questions.get(language, questions["en"]).get(field_name, questions["en"]["goal"])


def _parse_rest_days(text: str) -> list[str]:
    lowered = text.lower()
    english_map = {
        "saturday": "Saturday",
        "sunday": "Sunday",
        "monday": "Monday",
        "tuesday": "Tuesday",
        "wednesday": "Wednesday",
        "thursday": "Thursday",
        "friday": "Friday",
    }
    arabic_map = {
        "السبت": "Saturday",
        "الأحد": "Sunday",
        "الاحد": "Sunday",
        "الاثنين": "Monday",
        "الثلاثاء": "Tuesday",
        "الأربعاء": "Wednesday",
        "الاربعاء": "Wednesday",
        "الخميس": "Thursday",
        "الجمعة": "Friday",
    }

    results: list[str] = []
    for name, normalized in english_map.items():
        if name in lowered:
            results.append(normalized)
    for name, normalized in arabic_map.items():
        if name in text:
            results.append(normalized)

    deduped: list[str] = []
    for day_name in results:
        if day_name not in deduped:
            deduped.append(day_name)
    return deduped


def _apply_profile_answer(field_name: str, answer: str, user_state: dict[str, Any]) -> bool:
    text = answer.strip()
    lowered = text.lower()

    if field_name == "goal":
        normalized_goal = _normalize_goal(text)
        if not normalized_goal:
            return False
        user_state["goal"] = normalized_goal
        return True
    if field_name == "fitness_level":
        if "begin" in lowered or "مبت" in lowered:
            user_state["fitness_level"] = "beginner"
            return True
        if "inter" in lowered or "متوس" in lowered:
            user_state["fitness_level"] = "intermediate"
            return True
        if "adv" in lowered or "متقد" in lowered:
            user_state["fitness_level"] = "advanced"
            return True
        return False
    if field_name in {"age", "weight", "height"}:
        match = re.search(r"\d+(\.\d+)?", lowered)
        if not match:
            return False
        numeric_value = float(match.group())
        user_state[field_name] = int(numeric_value) if field_name == "age" else numeric_value
        return True
    if field_name == "gender":
        if any(token in lowered for token in ("male", "ذكر", "man")):
            user_state["gender"] = "male"
            return True
        if any(token in lowered for token in ("female", "أنث", "انث", "woman")):
            user_state["gender"] = "female"
            return True
        return False
    if field_name == "meals_per_day":
        match = re.search(r"\d+", lowered)
        if not match:
            return False
        meals_count = int(match.group())
        if meals_count < 3 or meals_count > 6:
            return False
        user_state["meals_per_day"] = meals_count
        return True
    if field_name == "rest_days":
        rest_days = _parse_rest_days(text)
        if not rest_days:
            return False
        user_state["rest_days"] = rest_days
        return True
    if field_name == "chronic_diseases":
        if any(token in lowered for token in ("none", "no", "لا يوجد", "ما في")):
            user_state["chronic_diseases"] = []
            return True
        user_state["chronic_diseases"] = _parse_list_field(text)
        return True
    if field_name == "allergies":
        if any(token in lowered for token in ("none", "no", "لا يوجد", "ما في")):
            user_state["allergies"] = []
            return True
        user_state["allergies"] = _parse_list_field(text)
        return True
    return False


def _select_exercises(focus: str, difficulty: str, max_items: int = 5) -> list[dict[str, Any]]:
    exercises: list[dict[str, Any]] = []
    allowed_difficulties = {
        "beginner": {"Beginner"},
        "intermediate": {"Beginner", "Intermediate"},
        "advanced": {"Beginner", "Intermediate", "Advanced"},
    }
    difficulty_filter = allowed_difficulties.get(difficulty, {"Beginner", "Intermediate"})

    for item in AI_ENGINE.exercises:
        muscle = str(item.get("muscle", "")).lower()
        level = str(item.get("difficulty", "Beginner"))
        if focus in muscle and level in difficulty_filter:
            exercises.append(item)
        if len(exercises) >= max_items:
            break

    if exercises:
        return exercises
    return AI_ENGINE.exercises[:max_items]


def _generate_workout_plan(profile: dict[str, Any], language: str) -> dict[str, Any]:
    goal = profile.get("goal") or "general_fitness"
    difficulty = str(profile.get("fitness_level", "beginner")).lower()
    rest_days = profile.get("rest_days") or ["Friday"]
    rest_days = [day for day in rest_days if isinstance(day, str)]

    if goal == "muscle_gain":
        weekly_focus = ["chest", "back", "legs", "shoulders", "core"]
        default_sets, default_reps = 4, "8-12"
    elif goal == "fat_loss":
        weekly_focus = ["legs", "core", "back", "chest", "shoulders"]
        default_sets, default_reps = 3, "12-15"
    else:
        weekly_focus = ["core", "legs", "back", "chest", "shoulders"]
        default_sets, default_reps = 3, "10-12"

    plan_days: list[dict[str, Any]] = []
    focus_index = 0
    for english_day, arabic_day in WEEK_DAYS:
        if english_day in rest_days:
            plan_days.append(
                {
                    "day": english_day,
                    "dayAr": arabic_day,
                    "focus": "Rest",
                    "exercises": [],
                }
            )
            continue

        focus = weekly_focus[focus_index % len(weekly_focus)]
        focus_index += 1
        exercise_items = _select_exercises(focus, difficulty, max_items=5)

        exercises = []
        for item in exercise_items:
            exercise_name = str(item.get("exercise", "Exercise"))
            exercises.append(
                {
                    "name": exercise_name,
                    "nameAr": exercise_name,
                    "sets": str(default_sets),
                    "reps": default_reps,
                    "rest_seconds": 90 if goal != "fat_loss" else 60,
                    "notes": str(item.get("description", "")),
                }
            )

        plan_days.append(
            {
                "day": english_day,
                "dayAr": arabic_day,
                "focus": focus,
                "exercises": exercises,
            }
        )

    title = "AI Workout Plan"
    title_ar = "خطة تمارين ذكية"
    if language == "ar_jordanian":
        title_ar = "خطة تمارين"

    return {
        "id": f"workout_{uuid.uuid4().hex[:10]}",
        "type": "workout",
        "title": title,
        "title_ar": title_ar,
        "goal": goal,
        "fitness_level": difficulty,
        "rest_days": rest_days,
        "duration_days": 7,
        "days": plan_days,
        "created_at": datetime.utcnow().isoformat(),
    }


def _calculate_calories(profile: dict[str, Any]) -> int:
    if profile.get("target_calories"):
        return int(profile["target_calories"])

    weight = float(profile.get("weight", 70))
    height = float(profile.get("height", 170))
    age = float(profile.get("age", 25))
    gender = str(profile.get("gender", "male")).lower()
    goal = str(profile.get("goal") or "general_fitness")
    fitness_level = str(profile.get("fitness_level", "beginner")).lower()

    bmr = 10 * weight + 6.25 * height - 5 * age + (5 if gender == "male" else -161)
    activity_factor = {"beginner": 1.40, "intermediate": 1.55, "advanced": 1.70}.get(fitness_level, 1.45)
    maintenance = bmr * activity_factor

    if goal == "muscle_gain":
        maintenance += 300
    elif goal == "fat_loss":
        maintenance -= 400

    return max(1200, int(round(maintenance)))


@lru_cache(maxsize=1)
def _allergy_categories_from_dataset() -> set[str]:
    candidates = [
        BACKEND_DIR / "datasets" / "food_allergy_dataset.csv",
        Path(r"D:\chatbot coach\Dataset\New folder\food_allergy_dataset.csv"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            import csv

            with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                values = {str(row.get("Food_Type", "")).strip().lower() for row in reader if row.get("Food_Type")}
                return {v for v in values if v}
        except Exception:
            continue
    return set()


ALLERGY_CATEGORY_TOKENS: dict[str, set[str]] = {
    "gluten": {
        "gluten",
        "wheat",
        "bread",
        "flour",
        "pasta",
        "oats",
        "barley",
        "rye",
        "قمح",
        "خبز",
        "طحين",
        "معكرونة",
        "شوفان",
        "شعير",
    },
    "dairy": {
        "milk",
        "cheese",
        "yogurt",
        "butter",
        "cream",
        "milk",
        "حليب",
        "جبن",
        "لبنة",
        "زبادي",
        "زبدة",
        "قشطة",
    },
    "eggs": {
        "egg",
        "eggs",
        "omelette",
        "بيض",
        "بياض",
        "صفار",
        "اومليت",
    },
    "nuts": {
        "nuts",
        "peanut",
        "almond",
        "walnut",
        "cashew",
        "hazelnut",
        "pistachio",
        "مكسرات",
        "فول سوداني",
        "لوز",
        "جوز",
        "كاجو",
        "بندق",
        "فستق",
    },
    "seafood": {
        "seafood",
        "fish",
        "salmon",
        "tuna",
        "shrimp",
        "crab",
        "lobster",
        "سمك",
        "سلمون",
        "تونة",
        "جمبري",
        "روبيان",
        "سرطان",
        "لوبستر",
    },
}

CHRONIC_RESTRICTION_TOKENS: dict[str, set[str]] = {
    "diabetes": {
        "sugar",
        "sweet",
        "sweets",
        "soda",
        "juice",
        "white bread",
        "white rice",
        "dessert",
        "cake",
        "chocolate",
        "honey",
        "jam",
        "سكر",
        "حلويات",
        "عصير",
        "مشروبات غازية",
        "خبز ابيض",
        "رز ابيض",
        "كيك",
        "شوكولاتة",
        "عسل",
        "مربى",
    },
    "hypertension": {
        "salt",
        "salty",
        "sodium",
        "pickle",
        "processed",
        "sausage",
        "chips",
        "soy sauce",
        "ملح",
        "مخللات",
        "لحوم مصنعة",
        "نقانق",
        "شيبس",
        "صلصة الصويا",
    },
    "heart": {
        "fried",
        "butter",
        "ghee",
        "cream",
        "fatty",
        "red meat",
        "bacon",
        "sausages",
        "cheese",
        "مقلي",
        "زبدة",
        "سمنة",
        "دهون",
        "لحمة دهنية",
        "نقانق",
        "جبن",
    },
    "cholesterol": {
        "fried",
        "butter",
        "ghee",
        "cream",
        "fatty",
        "red meat",
        "bacon",
        "sausages",
        "cheese",
        "مقلي",
        "زبدة",
        "سمنة",
        "دهون",
        "لحمة دهنية",
        "نقانق",
        "جبن",
    },
}


def _text_contains_any(text: str, tokens: set[str]) -> bool:
    normalized = normalize_text(text or "")
    if not normalized or not tokens:
        return False
    return any(token and normalize_text(token) in normalized for token in tokens)


def _build_food_restrictions(profile: dict[str, Any]) -> dict[str, Any]:
    allergies = _parse_list_field(profile.get("allergies"))
    chronic = _parse_list_field(profile.get("chronic_diseases"))

    tokens: set[str] = set()
    labels: list[str] = []

    known_categories = _allergy_categories_from_dataset()

    for allergy in allergies:
        norm = normalize_text(allergy)
        matched = False
        for key, key_tokens in ALLERGY_CATEGORY_TOKENS.items():
            if key in norm or any(normalize_text(tok) in norm for tok in key_tokens):
                tokens |= key_tokens
                if key not in labels:
                    labels.append(key)
                matched = True
        if not matched and norm:
            tokens.add(allergy)
            if allergy not in labels:
                labels.append(allergy)

    # If dataset categories exist, include them in labels when user mentions them.
    for category in known_categories:
        if category and any(category in normalize_text(a) for a in allergies):
            if category not in labels:
                labels.append(category)

    for disease in chronic:
        norm = normalize_text(disease)
        if "diab" in norm or "سكر" in norm:
            tokens |= CHRONIC_RESTRICTION_TOKENS["diabetes"]
            labels.append("diabetes")
            continue
        if "ضغط" in norm or "hypertension" in norm:
            tokens |= CHRONIC_RESTRICTION_TOKENS["hypertension"]
            labels.append("hypertension")
            continue
        if "قلب" in norm or "heart" in norm:
            tokens |= CHRONIC_RESTRICTION_TOKENS["heart"]
            labels.append("heart")
            continue
        if "كوليسترول" in norm or "cholesterol" in norm:
            tokens |= CHRONIC_RESTRICTION_TOKENS["cholesterol"]
            labels.append("cholesterol")

    return {
        "tokens": {t for t in tokens if t},
        "labels": labels,
        "allergies": allergies,
        "chronic_diseases": chronic,
    }


def _filter_meals_by_restrictions(meals: list[dict[str, Any]], restriction_tokens: set[str]) -> list[dict[str, Any]]:
    if not meals or not restriction_tokens:
        return meals
    filtered: list[dict[str, Any]] = []
    for meal in meals:
        haystack = " ".join(
            [
                str(meal.get("meal_type", "")),
                str(meal.get("description", "")),
                str(meal.get("name", "")),
                str(meal.get("descriptionAr", "")),
                str(meal.get("nameAr", "")),
            ]
        )
        if _text_contains_any(haystack, restriction_tokens):
            continue
        filtered.append(meal)
    return filtered if filtered else meals


def _safe_meal_templates(allergies: list[str], restriction_tokens: set[str] | None = None) -> list[dict[str, Any]]:
    templates = [
        {"name": "Greek Yogurt + Oats + Berries", "calories": 420, "protein": 28, "carbs": 48, "fat": 12, "ingredients": ["yogurt", "oats", "berries"]},
        {"name": "Egg Omelette + Whole Grain Bread", "calories": 460, "protein": 32, "carbs": 34, "fat": 20, "ingredients": ["egg", "bread", "vegetables"]},
        {"name": "Chicken Rice Bowl", "calories": 620, "protein": 45, "carbs": 70, "fat": 16, "ingredients": ["chicken", "rice", "vegetables"]},
        {"name": "Salmon + Sweet Potato", "calories": 650, "protein": 42, "carbs": 58, "fat": 24, "ingredients": ["salmon", "sweet potato", "vegetables"]},
        {"name": "Tuna Wrap", "calories": 480, "protein": 35, "carbs": 44, "fat": 14, "ingredients": ["tuna", "whole wheat tortilla", "vegetables"]},
        {"name": "Lean Beef + Quinoa", "calories": 640, "protein": 43, "carbs": 55, "fat": 20, "ingredients": ["beef", "quinoa", "salad"]},
        {"name": "Protein Shake + Banana", "calories": 320, "protein": 30, "carbs": 34, "fat": 6, "ingredients": ["whey", "banana", "milk"]},
        {"name": "Cottage Cheese + Fruit", "calories": 300, "protein": 24, "carbs": 28, "fat": 8, "ingredients": ["cottage cheese", "fruit"]},
    ]

    allergy_tokens = {a.lower() for a in allergies}
    if restriction_tokens:
        allergy_tokens |= {t.lower() for t in restriction_tokens}
    safe: list[dict[str, Any]] = []
    for meal in templates:
        ingredients_text = " ".join(meal["ingredients"]).lower()
        if any(token and token in ingredients_text for token in allergy_tokens):
            continue
        safe.append(meal)
    return safe if safe else templates


def _build_nutrition_days(profile: dict[str, Any], calories_target: int) -> tuple[list[dict[str, Any]], int]:
    meals_per_day = int(profile.get("meals_per_day", 4))
    meals_per_day = max(3, min(6, meals_per_day))
    allergies = _parse_list_field(profile.get("allergies"))
    chronic = [d.lower() for d in _parse_list_field(profile.get("chronic_diseases"))]
    restrictions = _build_food_restrictions(profile)

    meal_templates = _safe_meal_templates(allergies, restrictions.get("tokens", set()))
    meal_templates.sort(key=lambda m: m["calories"])

    if any("diab" in x or "سكر" in x for x in chronic):
        for meal in meal_templates:
            meal["carbs"] = int(round(meal["carbs"] * 0.85))

    meal_ratio = [0.25, 0.10, 0.30, 0.10, 0.20, 0.05]
    day_plans: list[dict[str, Any]] = []
    total_protein = 0

    for day_index, (english_day, arabic_day) in enumerate(WEEK_DAYS):
        meals_for_day: list[dict[str, Any]] = []
        for i in range(meals_per_day):
            template = meal_templates[(i + day_index) % len(meal_templates)]
            target = int(calories_target * meal_ratio[i])
            scale = max(0.6, min(1.6, target / template["calories"]))

            calories = int(round(template["calories"] * scale))
            protein = int(round(template["protein"] * scale))
            carbs = int(round(template["carbs"] * scale))
            fat = int(round(template["fat"] * scale))

            total_protein += protein
            meals_for_day.append(
                {
                    "name": template["name"],
                    "nameAr": template["name"],
                    "description": f"Ingredients: {', '.join(template['ingredients'])}",
                    "descriptionAr": f"المكونات: {', '.join(template['ingredients'])}",
                    "calories": str(calories),
                    "protein": protein,
                    "carbs": carbs,
                    "fat": fat,
                    "time": f"meal_{i + 1}",
                }
            )

        day_plans.append({"day": english_day, "dayAr": arabic_day, "meals": meals_for_day})

    avg_daily_protein = int(round(total_protein / 7))
    return day_plans, avg_daily_protein


def _generate_nutrition_plan(profile: dict[str, Any], language: str) -> dict[str, Any]:
    calories_target = _calculate_calories(profile)
    days, avg_daily_protein = _build_nutrition_days(profile, calories_target)
    chronic = _parse_list_field(profile.get("chronic_diseases"))
    allergies = _parse_list_field(profile.get("allergies"))
    restrictions = _build_food_restrictions(profile)
    kb_query_parts = [
        "nutrition meal plan",
        str(profile.get("goal", "") or ""),
        " ".join(chronic),
        " ".join(allergies),
    ]
    kb_query = " ".join(part for part in kb_query_parts if part).strip()
    kb_hits = NUTRITION_KB.search(kb_query, top_k=2, max_chars=220) if NUTRITION_KB.ready and kb_query else []
    reference_notes = [hit["text"] for hit in kb_hits]

    notes = []
    if chronic:
        notes.append(f"Adjusted for chronic conditions: {', '.join(chronic)}.")
    if allergies:
        notes.append(f"Avoided allergens: {', '.join(allergies)}.")
    if restrictions.get("labels"):
        notes.append(f"Restricted foods based on profile: {', '.join(restrictions['labels'])}.")

    title = "AI Nutrition Plan"
    title_ar = "خطة تغذية ذكية"
    if language == "ar_jordanian":
        title_ar = "خطة أكل"

    return {
        "id": f"nutrition_{uuid.uuid4().hex[:10]}",
        "type": "nutrition",
        "title": title,
        "title_ar": title_ar,
        "goal": profile.get("goal", "general_fitness"),
        "daily_calories": calories_target,
        "estimated_protein": avg_daily_protein,
        "meals_per_day": int(profile.get("meals_per_day", 4)),
        "days": days,
        "notes": " ".join(notes).strip(),
        "forbidden_foods": list(restrictions.get("labels", [])),
        "reference_notes": reference_notes,
        "created_at": datetime.utcnow().isoformat(),
    }


def _format_plan_preview(plan_type: str, plan: dict[str, Any], language: str) -> str:
    if plan_type == "workout":
        workout_days = [d for d in plan.get("days", []) if d.get("exercises")]
        rest_days = [d.get("day") for d in plan.get("days", []) if not d.get("exercises")]
        sample = workout_days[0]["exercises"][:3] if workout_days else []
        sample_text = "\n".join([f"- {x['name']} ({x['sets']}x{x['reps']})" for x in sample])

        if language == "en":
            return (
                f"I prepared a 7-day workout plan for your goal.\n"
                f"Rest days: {', '.join(rest_days) if rest_days else 'None'}\n"
                f"Sample day:\n{sample_text}\n\n"
                "Do you want to approve this plan and add it to your schedule page?"
            )
        if language == "ar_fusha":
            return (
                f"أعددت لك خطة تمارين لمدة 7 أيام حسب هدفك.\n"
                f"أيام الراحة: {', '.join(rest_days) if rest_days else 'لا يوجد'}\n"
                f"مثال ليوم تدريبي:\n{sample_text}\n\n"
                "هل تريد اعتماد هذه الخطة وإضافتها إلى صفحة الجدول؟"
            )
        return (
            f"جهزتلك خطة تمارين 7 أيام حسب هدفك.\n"
            f"أيام الراحة: {', '.join(rest_days) if rest_days else 'ما في'}\n"
            f"مثال يوم تدريبي:\n{sample_text}\n\n"
            "بدك تعتمد الخطة وتنزل مباشرة بصفحة الجدول؟"
        )

    calories = plan.get("daily_calories", 0)
    meals_count = plan.get("meals_per_day", 4)
    sample_meals = plan.get("days", [{}])[0].get("meals", [])[:3]
    sample_text = "\n".join([f"- {m['name']} ({m['calories']} kcal)" for m in sample_meals])

    if language == "en":
        return (
            f"I prepared a nutrition plan: {calories} kcal/day, {meals_count} meals/day.\n"
            f"Sample meals:\n{sample_text}\n\n"
            "Do you want to approve this plan and add it to your schedule page?"
        )
    if language == "ar_fusha":
        return (
            f"أعددت لك خطة غذائية: {calories} سعرة يوميًا، {meals_count} وجبات يوميًا.\n"
            f"عينة من الوجبات:\n{sample_text}\n\n"
            "هل تريد اعتماد هذه الخطة وإضافتها إلى صفحة الجدول؟"
        )
    return (
        f"جهزتلك خطة أكل: {calories} سعرة باليوم، {meals_count} وجبات باليوم.\n"
        f"عينة وجبات:\n{sample_text}\n\n"
        "بدك تعتمدها وتنزل على صفحة الجدول؟"
    )


def _generate_workout_plan_options(profile: dict[str, Any], language: str, count: int = 5) -> list[dict[str, Any]]:
    dataset_options = _generate_workout_plan_options_from_dataset(profile, language, count)
    return dataset_options

    variants = [
        {
            "key": "balanced_strength",
            "title": "Balanced Strength Split",
            "title_ar": "خطة قوة متوازنة",
            "focus_cycle": ["legs", "chest", "back", "shoulders", "core"],
            "sets": "4",
            "reps": "8-12",
            "rest_seconds": 90,
            "exercise_count": 5,
        },
        {
            "key": "upper_lower",
            "title": "Upper / Lower Split",
            "title_ar": "خطة علوي وسفلي",
            "focus_cycle": ["chest", "legs", "back", "legs", "shoulders"],
            "sets": "4",
            "reps": "6-10",
            "rest_seconds": 120,
            "exercise_count": 4,
        },
        {
            "key": "hypertrophy_volume",
            "title": "Hypertrophy Volume",
            "title_ar": "خطة تضخيم حجم",
            "focus_cycle": ["chest", "back", "legs", "shoulders", "arms"],
            "sets": "5",
            "reps": "10-15",
            "rest_seconds": 75,
            "exercise_count": 5,
        },
        {
            "key": "fat_loss_circuit",
            "title": "Fat Loss Circuit",
            "title_ar": "خطة حرق دهون دائرية",
            "focus_cycle": ["legs", "core", "back", "chest", "full body"],
            "sets": "3",
            "reps": "12-20",
            "rest_seconds": 45,
            "exercise_count": 6,
        },
        {
            "key": "beginner_foundation",
            "title": "Beginner Foundation",
            "title_ar": "خطة تأسيس للمبتدئ",
            "focus_cycle": ["full body", "legs", "back", "chest", "core"],
            "sets": "3",
            "reps": "10-12",
            "rest_seconds": 75,
            "exercise_count": 4,
        },
        {
            "key": "athletic_performance",
            "title": "Athletic Performance",
            "title_ar": "خطة أداء رياضي",
            "focus_cycle": ["legs", "back", "core", "shoulders", "full body"],
            "sets": "4",
            "reps": "6-10",
            "rest_seconds": 90,
            "exercise_count": 5,
        },
    ]

    if str(profile.get("fitness_level", "")).lower() == "beginner":
        variants = sorted(variants, key=lambda v: 0 if v["key"] == "beginner_foundation" else 1)
    if str(profile.get("goal", "")).lower() == "fat_loss":
        variants = sorted(variants, key=lambda v: 0 if v["key"] == "fat_loss_circuit" else 1)

    selected_variants = variants[: max(1, min(count, len(variants)))]
    rest_days = [d for d in profile.get("rest_days", ["Friday"]) if isinstance(d, str)]
    difficulty = str(profile.get("fitness_level", "beginner")).lower()

    options: list[dict[str, Any]] = []
    for variant in selected_variants:
        plan = _generate_workout_plan(profile, language)
        plan["id"] = f"workout_{uuid.uuid4().hex[:10]}"
        plan["title"] = variant["title"]
        plan["title_ar"] = variant["title_ar"]
        plan["rest_days"] = rest_days
        plan_days: list[dict[str, Any]] = []
        focus_index = 0

        for english_day, arabic_day in WEEK_DAYS:
            if english_day in rest_days:
                plan_days.append({"day": english_day, "dayAr": arabic_day, "focus": "Rest", "exercises": []})
                continue

            focus = variant["focus_cycle"][focus_index % len(variant["focus_cycle"])]
            focus_index += 1
            exercise_items = _select_exercises(focus, difficulty, max_items=int(variant["exercise_count"]))
            exercises = [
                {
                    "name": str(item.get("exercise", "Exercise")),
                    "nameAr": str(item.get("exercise", "Exercise")),
                    "sets": variant["sets"],
                    "reps": variant["reps"],
                    "rest_seconds": int(variant["rest_seconds"]),
                    "notes": str(item.get("description", "")),
                }
                for item in exercise_items
            ]
            plan_days.append({"day": english_day, "dayAr": arabic_day, "focus": focus, "exercises": exercises})

        plan["days"] = plan_days
        plan["variant_key"] = variant["key"]
        options.append(plan)
    return options


def _generate_nutrition_plan_options(profile: dict[str, Any], language: str, count: int = 5) -> list[dict[str, Any]]:
    dataset_options = _generate_nutrition_plan_options_from_dataset(profile, language, count)
    return dataset_options

    styles = [
        {"key": "balanced", "title": "Balanced Daily Nutrition", "title_ar": "خطة تغذية متوازنة", "calorie_shift": 0, "protein_mul": 1.00, "carb_mul": 1.00, "fat_mul": 1.00},
        {"key": "high_protein", "title": "High Protein Plan", "title_ar": "خطة بروتين عالي", "calorie_shift": 80, "protein_mul": 1.20, "carb_mul": 0.90, "fat_mul": 0.95},
        {"key": "cutting_lean", "title": "Lean Cutting Plan", "title_ar": "خطة تنشيف", "calorie_shift": -180, "protein_mul": 1.15, "carb_mul": 0.80, "fat_mul": 0.90},
        {"key": "mass_gain", "title": "Mass Gain Plan", "title_ar": "خطة زيادة كتلة", "calorie_shift": 220, "protein_mul": 1.10, "carb_mul": 1.20, "fat_mul": 1.05},
        {"key": "low_gi", "title": "Low GI Plan", "title_ar": "خطة مؤشر سكري منخفض", "calorie_shift": -60, "protein_mul": 1.05, "carb_mul": 0.85, "fat_mul": 1.00},
        {"key": "budget", "title": "Budget Friendly Plan", "title_ar": "خطة اقتصادية", "calorie_shift": 0, "protein_mul": 1.00, "carb_mul": 1.05, "fat_mul": 0.95},
    ]

    goal = str(profile.get("goal", "")).lower()
    if goal == "fat_loss":
        styles = sorted(styles, key=lambda s: 0 if s["key"] == "cutting_lean" else 1)
    elif goal == "muscle_gain":
        styles = sorted(styles, key=lambda s: 0 if s["key"] == "mass_gain" else 1)

    selected_styles = styles[: max(1, min(count, len(styles)))]
    options: list[dict[str, Any]] = []
    for style in selected_styles:
        plan = _generate_nutrition_plan(profile, language)
        plan["id"] = f"nutrition_{uuid.uuid4().hex[:10]}"
        plan["title"] = style["title"]
        plan["title_ar"] = style["title_ar"]
        plan["daily_calories"] = max(1200, int(plan.get("daily_calories", 2000) + style["calorie_shift"]))

        new_days = []
        for day in plan.get("days", []):
            meals = []
            for meal in day.get("meals", []):
                protein = max(5, int(round(float(meal.get("protein", 0)) * style["protein_mul"])))
                carbs = max(5, int(round(float(meal.get("carbs", 0)) * style["carb_mul"])))
                fat = max(3, int(round(float(meal.get("fat", 0)) * style["fat_mul"])))
                calories = int(round((protein * 4) + (carbs * 4) + (fat * 9)))
                meals.append(
                    {
                        **meal,
                        "protein": protein,
                        "carbs": carbs,
                        "fat": fat,
                        "calories": str(calories),
                    }
                )
            new_days.append({**day, "meals": meals})

        plan["days"] = new_days
        plan["variant_key"] = style["key"]
        options.append(plan)
    return options


def _format_plan_options_preview(plan_type: str, options: list[dict[str, Any]], language: str) -> str:
    if not options:
        if language == "en":
            return "I could not generate options right now. Please retry."
        if language == "ar_fusha":
            return "تعذر توليد خيارات الآن. حاول مرة أخرى."
        return "ما قدرت اولّد خيارات هسا. جرّب مرة ثانية."

    lines = []
    for i, plan in enumerate(options, start=1):
        if plan_type == "workout":
            rest_days = ", ".join(plan.get("rest_days", [])) or "None"
            sample_focus = next((d.get("focus") for d in plan.get("days", []) if d.get("exercises")), "general")
            lines.append(f"{i}. {plan.get('title', 'Workout Plan')} | focus: {sample_focus} | rest: {rest_days}")
        else:
            lines.append(
                f"{i}. {plan.get('title', 'Nutrition Plan')} | "
                f"{plan.get('daily_calories', 0)} kcal/day | {plan.get('meals_per_day', 4)} meals/day"
            )

    options_text = "\n".join(lines)
    if language == "en":
        return (
            "I prepared multiple options for you:\n"
            f"{options_text}\n\n"
            "Reply with the option number you want (for example: 1)."
        )
    if language == "ar_fusha":
        return (
            "أعددت لك عدة خيارات:\n"
            f"{options_text}\n\n"
            "أرسل رقم الخيار الذي تريده (مثال: 1)."
        )
    return (
        "جهزتلك كذا خيار:\n"
        f"{options_text}\n\n"
        "ابعت رقم الخيار اللي بدك ياه (مثال: 1)."
    )


def _extract_plan_choice_index(user_input: str, options_count: int) -> int | None:
    if options_count <= 0:
        return None

    number = extract_first_int(user_input)
    if number is not None and 1 <= number <= options_count:
        return number - 1

    normalized = normalize_text(user_input)
    word_to_index = {
        "first": 0,
        "second": 1,
        "third": 2,
        "fourth": 3,
        "fifth": 4,
        "اول": 0,
        "ثاني": 1,
        "ثالث": 2,
        "رابع": 3,
        "خامس": 4,
    }
    for word, idx in word_to_index.items():
        if idx < options_count and fuzzy_contains_any(normalized, {word}):
            return idx
    return None


def _greeting_reply(language: str, profile: Optional[dict[str, Any]] = None) -> str:
    display_name = _profile_display_name(profile or {})
    dataset_reply = _dataset_intent_response("greeting", language, seed=display_name or "user")
    if dataset_reply:
        return dataset_reply

    name_suffix = f" {display_name}" if display_name else ""
    warmup = _motivation_line(language, f"greet-{display_name or 'user'}")
    if language == "en":
        return (
            f"Hi{name_suffix}! {warmup} "
            "I am your AI fitness coach. "
            "I can help with workouts, nutrition plans, and progress tracking."
        )
    if language == "ar_fusha":
        if display_name:
            return (
                f"مرحبًا {display_name}! {warmup} "
                "أنا مدربك الرياضي الذكي. "
                "أساعدك في التمارين والتغذية ومتابعة الالتزام."
            )
        return f"مرحبًا! {warmup} أنا مدربك الرياضي الذكي. أساعدك في التمارين والتغذية ومتابعة الالتزام."
    if display_name:
        return f"هلا {display_name}! {warmup} أنا كوتشك الذكي، وبساعدك بالتمارين والأكل ومتابعة الالتزام."
    return f"هلا! {warmup} أنا كوتشك الذكي، وبساعدك بالتمارين والأكل ومتابعة الالتزام."


def _name_reply(language: str) -> str:
    if language == "en":
        return "I am your AI Fitness Coach specialized in training and nutrition only."
    if language == "ar_fusha":
        return "أنا مدرب اللياقة الذكي الخاص بك، ومتخصص فقط في التدريب والتغذية."
    return "أنا كوتشك الذكي، ومتخصص بس بالتمارين والتغذية."


def _how_are_you_reply(language: str) -> str:
    if language == "en":
        return "I am ready to coach you. Tell me your goal and I will build your plan."
    if language == "ar_fusha":
        return "أنا جاهز لتدريبك. أخبرني بهدفك وسأبني لك خطة مناسبة."
    return "تمام وجاهز أدربك. احكيلي هدفك وببني لك خطة مناسبة."


def _exercise_reply(query: str, language: str) -> str:
    normalized = normalize_text(query)
    mapped_query = query
    muscle_map = {
        "صدر": "chest",
        "ظهر": "back",
        "كتف": "shoulders",
        "اكتاف": "shoulders",
        "ذراع": "arms",
        "باي": "biceps",
        "تراي": "triceps",
        "ارجل": "legs",
        "رجل": "legs",
        "ساق": "legs",
        "بطن": "core",
    }
    for ar_term, en_term in muscle_map.items():
        if ar_term in normalized:
            mapped_query = f"{en_term} workout"
            break

    results = AI_ENGINE.search_exercises(mapped_query, top_k=5)
    if not results:
        if language == "en":
            return "I could not find matching exercises. Rephrase your request and I will try again."
        if language == "ar_fusha":
            return "لم أجد تمارين مطابقة. أعد صياغة طلبك وسأحاول مرة أخرى."
        return "ما لقيت تمارين مطابقة. جرّب صياغة ثانية وبرجع بدور."

    lines = []
    for item in results:
        lines.append(
            f"- {item.get('exercise')} | {item.get('muscle')} | {item.get('difficulty')}\n"
            f"  {item.get('description')}"
        )

    if language == "en":
        suffix = "\nYou can view muscle-specific exercises in the app on: /workouts (3D muscle viewer)."
    elif language == "ar_fusha":
        suffix = "\nيمكنك مشاهدة تمارين كل عضلة داخل التطبيق عبر صفحة: /workouts (المجسم العضلي)."
    else:
        suffix = "\nبتقدر تشوف تمارين كل عضلة داخل التطبيق بصفحة: /workouts (المجسم)."

    return "\n".join(lines) + suffix


def _tracking_reply(language: str, tracking_summary: Optional[dict[str, Any]]) -> str:
    if not tracking_summary:
        if language == "en":
            return (
                f"{_motivation_line(language, 'tracking-empty')} "
                "I do not have your latest tracking snapshot yet. Keep checking tasks in Schedule and I will monitor your adherence."
            )
        if language == "ar_fusha":
            return (
                f"{_motivation_line(language, 'tracking-empty')} "
                "لا أملك حالياً آخر ملخص متابعة لك. استمر بتحديد المهام في صفحة الجدول وسأتابع التزامك."
            )
        return (
            f"{_motivation_line(language, 'tracking-empty')} "
            "لسا ما وصلني آخر ملخص متابعة. ضل علّم المهام بصفحة الجدول وأنا براقب التزامك."
        )

    completed = int(tracking_summary.get("completed_tasks", 0))
    total = int(tracking_summary.get("total_tasks", 0))
    adherence = float(tracking_summary.get("adherence_score", 0))
    adherence_pct = int(round(adherence * 100))

    if language == "en":
        return (
            f"{_motivation_line(language, f'track-{completed}-{total}')} "
            f"Progress update: {completed}/{total} tasks done, adherence {adherence_pct}%.\n"
            "Based on your recent tracking, keep this consistency. If you want, I can adjust your plan intensity for next week."
        )
    if language == "ar_fusha":
        return (
            f"{_motivation_line(language, f'track-{completed}-{total}')} "
            f"تحديث التقدم: أنجزت {completed}/{total} مهمة، ونسبة الالتزام {adherence_pct}%.\n"
            "حسب تقدمك الأسبوع الماضي، استمر على هذا النسق، ويمكنني تعديل شدة الخطة للأسبوع القادم إذا أردت."
        )
    return (
        f"{_motivation_line(language, f'track-{completed}-{total}')} "
        f"تحديث الإنجاز: خلصت {completed}/{total} مهمة، والتزامك {adherence_pct}%.\n"
        "حسب تقدمك الأسبوع الماضي، استمر هيك، وإذا بدك بقدر أعدل شدة الخطة للأسبوع الجاي."
    )


def _dict_get_any(source: Any, keys: list[str]) -> Any:
    if not isinstance(source, dict):
        return None
    for key in keys:
        if key in source and source[key] not in (None, ""):
            return source[key]
    return None


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _deep_merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base or {})
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _extract_json_objects(text: str) -> list[str]:
    results: list[str] = []
    start_idx: Optional[int] = None
    depth = 0
    for idx, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_idx is not None:
                candidate = text[start_idx : idx + 1].strip()
                if candidate:
                    results.append(candidate)
                start_idx = None
    return results


def _try_parse_json_object(raw_text: str) -> Optional[dict[str, Any]]:
    candidate = (raw_text or "").strip()
    if not candidate:
        return None

    parse_candidates = [
        candidate,
        re.sub(r",\s*([}\]])", r"\1", candidate),
    ]
    for payload in parse_candidates:
        try:
            obj = json.loads(payload)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _looks_like_tracking_summary(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if any(key in payload for key in ("goal", "weekly_stats", "monthly_stats", "adherence_score")):
        return True
    # Some payloads may arrive flattened.
    flat_keys = {"goal.type", "goal.current_weight", "goal.target_weight", "weekly_stats.weight_change"}
    return any(key in payload for key in flat_keys)


def _extract_float_from_patterns(source: str, patterns: list[str]) -> Optional[float]:
    for pattern in patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if not match:
            continue
        parsed = _to_float(match.group(1))
        if parsed is not None:
            return parsed
    return None


def _extract_float_series_from_patterns(source: str, patterns: list[str]) -> list[float]:
    for pattern in patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if not match:
            continue
        values = _to_float_list(match.group(1))
        if len(values) >= 2:
            return values
    return []


def _extract_goal_type_from_patterns(source: str) -> str:
    goal_patterns = [
        r"(?:goal(?:\s*type)?|goal_type|نوع\s*الهدف|الهدف)\s*[:=]\s*([a-z_\-\s\u0600-\u06FF]+)",
    ]
    for pattern in goal_patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if not match:
            continue
        normalized = _normalize_goal(match.group(1))
        if normalized in {"muscle_gain", "fat_loss", "general_fitness"}:
            return normalized
    inferred = _normalize_goal(source)
    if inferred in {"muscle_gain", "fat_loss", "general_fitness"}:
        return inferred
    return ""


def _extract_tracking_summary_from_message(
    user_input: str,
    profile: dict[str, Any],
) -> Optional[dict[str, Any]]:
    source = _repair_mojibake(user_input or "")
    if not source:
        return None

    extracted: dict[str, Any] = {}
    has_tracking_signal = False

    for candidate in _extract_json_objects(source):
        obj = _try_parse_json_object(candidate)
        if not obj:
            continue
        if _looks_like_tracking_summary(obj):
            extracted = _deep_merge_dict(extracted, obj)
            has_tracking_signal = True

    goal_payload = extracted.get("goal") if isinstance(extracted.get("goal"), dict) else {}
    weekly_payload = extracted.get("weekly_stats") if isinstance(extracted.get("weekly_stats"), dict) else {}
    monthly_payload = extracted.get("monthly_stats") if isinstance(extracted.get("monthly_stats"), dict) else {}

    goal_type = _extract_goal_type_from_patterns(source)
    if goal_type:
        goal_payload["type"] = goal_type
        has_tracking_signal = True

    number_pattern = r"([+-]?\d+(?:\.\d+)?)(?:\s*\+)?"
    current_weight = _extract_float_from_patterns(
        source,
        [
            rf"(?:current[_\s-]*weight|weight[_\s-]*now|وزن(?:ي)?\s*(?:الحالي|الان|الآن)?)\s*[:=]?\s*{number_pattern}",
            rf"(?:وزن(?:ي)?|وزني)\s*[:=]?\s*{number_pattern}",
            rf"(?:goal\.current_weight|current_weight)\s*[:=]?\s*{number_pattern}",
        ],
    )
    target_weight = _extract_float_from_patterns(
        source,
        [
            rf"(?:target[_\s-]*weight|goal[_\s-]*weight|الوزن\s*(?:المستهدف|الهدف)|هدف(?:ي)?\s*وزن)\s*[:=]?\s*{number_pattern}",
            rf"(?:هدفي|هدف(?:ي)?)\s*[:=]?\s*{number_pattern}",
            rf"(?:goal\.target_weight|target_weight)\s*[:=]?\s*{number_pattern}",
        ],
    )
    weekly_weight_change = _extract_float_from_patterns(
        source,
        [
            rf"(?:weekly[_\s-]*weight[_\s-]*change|weekly[_\s-]*change|تغير\s*الوزن\s*(?:الاسبوعي|الأسبوعي)|نزول\s*(?:اسبوعي|أسبوعي)|زيادة\s*(?:اسبوعية|أسبوعية))\s*[:=]?\s*{number_pattern}",
            rf"(?:weekly_stats\.weight_change|weight_change)\s*[:=]?\s*{number_pattern}",
        ],
    )

    if weekly_weight_change is None:
        gain_match = re.search(
            rf"(?:زاد(?:ت)?\s*وزن(?:ي)?|وزن(?:ي)?\s*زاد|وزن(?:ي)?\s*بزيد|وزن(?:ي)?\s*عم\s*يزيد|زيادة\s*وزن(?:ي)?)\s*(?:بالاسبوع|بالأسبوع|اسبوعي|أسبوعي)?\s*[:=]?\s*{number_pattern}",
            source,
            flags=re.IGNORECASE,
        )
        loss_match = re.search(
            rf"(?:نقص(?:ت)?\s*وزن(?:ي)?|وزن(?:ي)?\s*نقص|وزن(?:ي)?\s*بنقص|وزن(?:ي)?\s*عم\s*ينقص|نزول\s*وزن(?:ي)?|خسرت\s*وزن(?:ي)?)\s*(?:بالاسبوع|بالأسبوع|اسبوعي|أسبوعي)?\s*[:=]?\s*{number_pattern}",
            source,
            flags=re.IGNORECASE,
        )
        if gain_match:
            weekly_weight_change = _to_float(gain_match.group(1))
        elif loss_match:
            loss_value = _to_float(loss_match.group(1))
            weekly_weight_change = -abs(loss_value) if loss_value is not None else None
    monthly_weight_change = _extract_float_from_patterns(
        source,
        [
            rf"(?:monthly[_\s-]*weight[_\s-]*change|monthly[_\s-]*change|تغير\s*الوزن\s*الشهري)\s*[:=]?\s*{number_pattern}",
            rf"(?:monthly_stats\.weight_change|monthly_weight_change)\s*[:=]?\s*{number_pattern}",
        ],
    )
    strength_increase = _extract_float_from_patterns(
        source,
        [
            rf"(?:strength[_\s-]*increase(?:[_\s-]*percent)?|strength[_\s-]*percent|زيادة\s*القوة(?:\s*الشهرية)?)\s*[:=]?\s*{number_pattern}\s*%?",
            rf"(?:monthly_stats\.strength_increase_percent|strength_increase_percent)\s*[:=]?\s*{number_pattern}",
        ],
    )
    consistency_percent = _extract_float_from_patterns(
        source,
        [
            rf"(?:consistency(?:[_\s-]*percent)?|consistency[_\s-]*pct|نسبة\s*الالتزام|الالتزام)\s*[:=]?\s*{number_pattern}\s*%?",
            rf"(?:monthly_stats\.consistency_percent|consistency_percent)\s*[:=]?\s*{number_pattern}",
        ],
    )
    workout_days = _extract_float_from_patterns(
        source,
        [
            rf"(?:workout[_\s-]*days|days[_\s-]*trained|ايام\s*التمرين|أيام\s*التمرين)\s*[:=]?\s*{number_pattern}",
            rf"(?:weekly_stats\.workout_days|workout_days)\s*[:=]?\s*{number_pattern}",
        ],
    )
    planned_days = _extract_float_from_patterns(
        source,
        [
            rf"(?:planned[_\s-]*days|plan[_\s-]*days|ايام\s*الخطة|أيام\s*الخطة)\s*[:=]?\s*{number_pattern}",
            rf"(?:weekly_stats\.planned_days|planned_days)\s*[:=]?\s*{number_pattern}",
        ],
    )
    avg_calories = _extract_float_from_patterns(
        source,
        [
            rf"(?:avg[_\s-]*calories|average[_\s-]*calories|متوسط\s*السعرات|السعرات)\s*[:=]?\s*{number_pattern}",
            rf"(?:weekly_stats\.avg_calories|avg_calories)\s*[:=]?\s*{number_pattern}",
        ],
    )
    avg_protein = _extract_float_from_patterns(
        source,
        [
            rf"(?:avg[_\s-]*protein|average[_\s-]*protein|متوسط\s*البروتين|البروتين)\s*[:=]?\s*{number_pattern}",
            rf"(?:weekly_stats\.avg_protein|avg_protein)\s*[:=]?\s*{number_pattern}",
        ],
    )
    sleep_avg_hours = _extract_float_from_patterns(
        source,
        [
            rf"(?:sleep[_\s-]*avg[_\s-]*hours|average[_\s-]*sleep|sleep[_\s-]*hours|متوسط\s*النوم|ساعات\s*النوم)\s*[:=]?\s*{number_pattern}",
            rf"(?:weekly_stats\.sleep_avg_hours|sleep_avg_hours)\s*[:=]?\s*{number_pattern}",
        ],
    )
    weight_change_history = _extract_float_series_from_patterns(
        source,
        [
            r"(?:weight[_\s-]*change[_\s-]*history|weekly[_\s-]*history|last[_\s-]*4[_\s-]*weeks(?:[_\s-]*weight[_\s-]*change)?)\s*[:=]\s*([0-9,\.\-\+\s|;/]+)",
            r"(?:تغير(?:ات)?\s*الوزن\s*(?:آخر|اخر)\s*4\s*(?:اسابيع|أسابيع)|آخر\s*4\s*(?:اسابيع|أسابيع)\s*تغير\s*الوزن)\s*[:=]\s*([0-9,\.\-\+\s|;/]+)",
        ],
    )

    if current_weight is not None:
        goal_payload["current_weight"] = current_weight
        has_tracking_signal = True
    if target_weight is not None:
        goal_payload["target_weight"] = target_weight
        has_tracking_signal = True
    if weekly_weight_change is not None:
        weekly_payload["weight_change"] = weekly_weight_change
        has_tracking_signal = True
    if monthly_weight_change is not None:
        monthly_payload["weight_change"] = monthly_weight_change
        has_tracking_signal = True
    if strength_increase is not None:
        monthly_payload["strength_increase_percent"] = strength_increase
        has_tracking_signal = True
    if consistency_percent is not None:
        monthly_payload["consistency_percent"] = consistency_percent
        has_tracking_signal = True
    if workout_days is not None:
        weekly_payload["workout_days"] = workout_days
        has_tracking_signal = True
    if planned_days is not None:
        weekly_payload["planned_days"] = planned_days
        has_tracking_signal = True
    if avg_calories is not None:
        weekly_payload["avg_calories"] = avg_calories
        has_tracking_signal = True
    if avg_protein is not None:
        weekly_payload["avg_protein"] = avg_protein
        has_tracking_signal = True
    if sleep_avg_hours is not None:
        weekly_payload["sleep_avg_hours"] = sleep_avg_hours
        has_tracking_signal = True
    if weight_change_history:
        weekly_payload["weight_change_history"] = weight_change_history[-4:]
        has_tracking_signal = True

    if goal_payload:
        extracted["goal"] = goal_payload
    if weekly_payload:
        extracted["weekly_stats"] = weekly_payload
    if monthly_payload:
        extracted["monthly_stats"] = monthly_payload

    if not has_tracking_signal:
        return None
    return extracted or None


def _merge_tracking_summaries(
    current_summary: Optional[dict[str, Any]],
    new_summary: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not isinstance(current_summary, dict) and not isinstance(new_summary, dict):
        return None
    if not isinstance(current_summary, dict):
        return deepcopy(new_summary) if isinstance(new_summary, dict) else None
    if not isinstance(new_summary, dict):
        return deepcopy(current_summary)
    return _deep_merge_dict(current_summary, new_summary)


def _has_actionable_tracking_metrics(summary: Optional[dict[str, Any]]) -> bool:
    if not isinstance(summary, dict):
        return False

    goal = summary.get("goal") if isinstance(summary.get("goal"), dict) else {}
    weekly = summary.get("weekly_stats") if isinstance(summary.get("weekly_stats"), dict) else {}
    monthly = summary.get("monthly_stats") if isinstance(summary.get("monthly_stats"), dict) else {}

    if _to_float(_dict_get_any(goal, ["current_weight", "target_weight", "target_strength_increase_percent"])) is not None:
        return True
    if _to_float(_dict_get_any(weekly, ["weight_change", "weekly_weight_change"])) is not None:
        return True
    if _to_float(_dict_get_any(monthly, ["strength_increase_percent", "weight_change"])) is not None:
        return True
    if _to_float(_dict_get_any(monthly, ["consistency_percent"])) is not None:
        return True
    if _to_float_list(_dict_get_any(weekly, ["weight_change_history", "weight_change_last_4_weeks"])):
        return True
    if _to_float_list(_dict_get_any(summary, ["weekly_weight_change_history", "last_4_weeks_weight_change"])):
        return True

    return False


def _is_performance_analysis_request(
    user_input: str,
    message_tracking_summary: Optional[dict[str, Any]] = None,
) -> bool:
    normalized = normalize_text(user_input)
    if not normalized:
        return False

    if _contains_any(normalized, PERFORMANCE_ANALYSIS_KEYWORDS):
        return True

    if _contains_any(normalized, {"analyze", "analysis", "حلل", "تحليل", "قيّم", "قيم"}):
        if _contains_any(normalized, {"performance", "progress", "اداء", "أداء", "ادائي", "تقدمي", "تقدم"}):
            return True

    intent_terms = {
        "analysis",
        "analyze",
        "progress rate",
        "on track",
        "ahead",
        "behind",
        "estimate",
        "timeline",
        "weeks remaining",
        "تحليل",
        "حلل",
        "تقييم",
        "على المسار",
        "متقدم",
        "متاخر",
        "متأخر",
        "كم اسبوع",
        "كم أسبوع",
        "الوقت المتبقي",
        "المتبقي",
    }
    metric_terms = {
        "weight",
        "strength",
        "calories",
        "protein",
        "sleep",
        "consistency",
        "وزن",
        "قوة",
        "سعرات",
        "بروتين",
        "نوم",
        "التزام",
        "تقدم",
    }
    if _contains_any(normalized, intent_terms) and _contains_any(normalized, metric_terms):
        return True

    # If the user sends actionable tracking metrics in the same message, treat it as analysis intent.
    if _has_actionable_tracking_metrics(message_tracking_summary):
        return True

    return False


def _format_number(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _to_float_list(value: Any) -> list[float]:
    values: list[float] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                parsed = _to_float(
                    _dict_get_any(item, ["weight_change", "weekly_weight_change", "weightChange", "delta", "change"])
                )
            else:
                parsed = _to_float(item)
            if parsed is not None:
                values.append(parsed)
        return values

    if isinstance(value, str):
        for token in re.findall(r"-?\d+(?:\.\d+)?", value):
            parsed = _to_float(token)
            if parsed is not None:
                values.append(parsed)
    return values


def _extract_weight_change_series(
    tracking_summary: dict[str, Any],
    weekly_stats: dict[str, Any],
) -> list[float]:
    direct_series_keys = [
        "weight_change_last_4_weeks",
        "weight_change_history",
        "last_4_weeks_weight_change",
        "weekly_weight_change_history",
        "last4_weight_change",
        "recent_weight_changes",
    ]
    for key in direct_series_keys:
        if key in weekly_stats:
            values = _to_float_list(weekly_stats.get(key))
            if values:
                return values

    summary_series_keys = [
        "weekly_weight_change_history",
        "weight_change_history",
        "last_4_weeks_weight_change",
        "recent_weight_changes",
        "last_4_weeks",
    ]
    for key in summary_series_keys:
        if key in tracking_summary:
            values = _to_float_list(tracking_summary.get(key))
            if values:
                return values

    weekly_history = tracking_summary.get("weekly_history")
    values = _to_float_list(weekly_history)
    if values:
        return values

    return []


def _average(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _mean_abs_deviation(values: list[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean_value = _average(values)
    if mean_value is None:
        return None
    return sum(abs(item - mean_value) for item in values) / len(values)


def _fitness_level_to_experience(value: Any) -> float:
    normalized = normalize_text(str(value or ""))
    if any(token in normalized for token in {"advanced", "adv", "متقدم"}):
        return 3.0
    if any(token in normalized for token in {"intermediate", "inter", "متوسط"}):
        return 2.0
    if any(token in normalized for token in {"beginner", "beg", "مبتد"}):
        return 1.0
    parsed = _to_float(value)
    return float(parsed) if parsed is not None else 0.0


def _is_goal_prediction_request(user_input: str) -> bool:
    normalized = normalize_text(user_input)
    if _contains_any(normalized, ML_GOAL_QUERY_KEYWORDS):
        return True
    return _contains_any(normalized, {"goal", "هدف"}) and _contains_any(normalized, ML_GENERAL_PREDICTION_KEYWORDS)


def _is_success_prediction_request(user_input: str) -> bool:
    normalized = normalize_text(user_input)
    if _contains_any(normalized, ML_SUCCESS_QUERY_KEYWORDS):
        return True
    return _contains_any(normalized, {"success", "نجاح", "التزام"}) and _contains_any(
        normalized, ML_GENERAL_PREDICTION_KEYWORDS
    )


def _build_goal_prediction_payload(
    profile: dict[str, Any], tracking_summary: Optional[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    tracking_summary = tracking_summary if isinstance(tracking_summary, dict) else {}
    weekly_stats = tracking_summary.get("weekly_stats") if isinstance(tracking_summary.get("weekly_stats"), dict) else {}
    monthly_stats = tracking_summary.get("monthly_stats") if isinstance(tracking_summary.get("monthly_stats"), dict) else {}

    age = _to_float(profile.get("age"))
    gender = str(profile.get("gender") or "Other")
    weight_kg = _to_float(_dict_get_any(profile, ["weight", "weight_kg"]))

    height_value = _to_float(_dict_get_any(profile, ["height", "height_cm", "height_m"]))
    height_cm: Optional[float] = None
    height_m: Optional[float] = None
    if height_value is not None:
        if height_value > 3:
            height_cm = height_value
            height_m = height_value / 100.0
        else:
            height_m = height_value
            height_cm = height_value * 100.0

    fat_percentage = _to_float(_dict_get_any(profile, ["fat_percentage", "body_fat_percentage", "body_fat"]))
    workout_frequency_days_week = _to_float(
        _dict_get_any(weekly_stats, ["workout_days", "training_days", "sessions", "completed_workouts"])
    )
    calories_burned = _to_float(
        _dict_get_any(weekly_stats, ["calories_burned", "avg_calories_burned", "calories_burned_avg"])
    )
    if calories_burned is None:
        calories_burned = _to_float(_dict_get_any(monthly_stats, ["avg_calories_burned", "calories_burned"]))
    avg_bpm = _to_float(_dict_get_any(weekly_stats, ["avg_bpm", "heart_rate_avg", "average_bpm"]))

    payload = {
        "age": age or 0.0,
        "gender": gender,
        "weight_kg": weight_kg or 0.0,
        "height_m": height_m,
        "height_cm": height_cm,
        "bmi": _to_float(_dict_get_any(profile, ["bmi"])) or 0.0,
        "fat_percentage": fat_percentage or 0.0,
        "workout_frequency_days_week": workout_frequency_days_week or 0.0,
        "experience_level": _fitness_level_to_experience(profile.get("fitness_level")),
        "calories_burned": calories_burned or 0.0,
        "avg_bpm": avg_bpm or 0.0,
    }

    missing_fields: list[str] = []
    if age is None:
        missing_fields.append("age")
    if weight_kg is None:
        missing_fields.append("weight")
    if height_value is None:
        missing_fields.append("height")

    return payload, missing_fields


def _build_success_prediction_payload(
    profile: dict[str, Any], tracking_summary: Optional[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    tracking_summary = tracking_summary if isinstance(tracking_summary, dict) else {}
    weekly_stats = tracking_summary.get("weekly_stats") if isinstance(tracking_summary.get("weekly_stats"), dict) else {}
    monthly_stats = tracking_summary.get("monthly_stats") if isinstance(tracking_summary.get("monthly_stats"), dict) else {}

    age = _to_float(profile.get("age"))
    gender = str(profile.get("gender") or "Other")
    membership_type = str(_dict_get_any(profile, ["membership_type", "membership", "plan_type"]) or "Unknown")
    workout_type = str(
        _dict_get_any(weekly_stats, ["workout_type", "main_workout_type"])
        or _dict_get_any(profile, ["workout_type", "preferred_workout_type"])
        or "General"
    )
    workout_duration_minutes = _to_float(
        _dict_get_any(
            weekly_stats,
            ["avg_workout_duration_minutes", "workout_duration_minutes", "session_duration_minutes", "duration_minutes"],
        )
    )
    if workout_duration_minutes is None:
        workout_duration_minutes = _to_float(_dict_get_any(monthly_stats, ["avg_workout_duration_minutes"]))
    calories_burned = _to_float(
        _dict_get_any(weekly_stats, ["calories_burned", "avg_calories_burned", "calories_burned_avg"])
    )
    if calories_burned is None:
        calories_burned = _to_float(_dict_get_any(monthly_stats, ["avg_calories_burned", "calories_burned"]))

    check_in_hour_value = _to_float(_dict_get_any(weekly_stats, ["check_in_hour", "avg_check_in_hour"]))
    check_in_hour = int(check_in_hour_value) if check_in_hour_value is not None else int(datetime.utcnow().hour)

    payload = {
        "age": age or 0.0,
        "gender": gender,
        "membership_type": membership_type,
        "workout_type": workout_type,
        "workout_duration_minutes": workout_duration_minutes or 0.0,
        "calories_burned": calories_burned or 0.0,
        "check_in_hour": check_in_hour,
    }

    missing_fields: list[str] = []
    if age is None:
        missing_fields.append("age")
    if workout_duration_minutes is None:
        missing_fields.append("weekly_stats.avg_workout_duration_minutes")
    if calories_burned is None:
        missing_fields.append("weekly_stats.calories_burned")

    return payload, missing_fields


def _ml_missing_fields_reply(language: str, prediction_type: str, missing_fields: list[str]) -> str:
    missing_text = ", ".join(missing_fields)
    if prediction_type == "goal":
        return _lang_reply(
            language,
            f"To run goal prediction, I still need: {missing_text}.",
            f"لتشغيل توقع الهدف، أحتاج هذه البيانات: {missing_text}.",
            f"عشان أشغّل توقع الهدف، لسا بحتاج: {missing_text}.",
        )
    return _lang_reply(
        language,
        f"To run success prediction, I still need: {missing_text}.",
        f"لتشغيل توقع النجاح، أحتاج هذه البيانات: {missing_text}.",
        f"عشان أشغّل توقع النجاح، لسا بحتاج: {missing_text}.",
    )


def _goal_label_from_prediction(value: Any, language: str) -> str:
    key = str(value or "").strip().lower()
    if key in {"muscle_gain", "fat_loss", "general_fitness"}:
        return _profile_goal_label(key, language)
    return str(value or "unknown")


def _ml_prediction_chat_response(
    user_input: str,
    language: str,
    profile: dict[str, Any],
    tracking_summary: Optional[dict[str, Any]],
) -> Optional[tuple[str, dict[str, Any]]]:
    want_goal = _is_goal_prediction_request(user_input)
    want_success = _is_success_prediction_request(user_input)

    if not want_goal and not want_success:
        return None

    reply_parts: list[str] = []
    payload: dict[str, Any] = {}

    if want_goal:
        goal_features, missing = _build_goal_prediction_payload(profile, tracking_summary)
        if missing:
            reply_parts.append(_ml_missing_fields_reply(language, "goal", missing))
        else:
            try:
                result = predict_goal(goal_features)
                predicted_goal = result.get("predicted_goal")
                predicted_goal_label = _goal_label_from_prediction(predicted_goal, language)
                confidence = None
                probabilities = result.get("probabilities") if isinstance(result.get("probabilities"), dict) else {}
                if predicted_goal in probabilities:
                    confidence = _to_float(probabilities.get(predicted_goal))

                goal_reply = _lang_reply(
                    language,
                    (
                        f"Goal prediction: {predicted_goal_label}"
                        + (f" (confidence {_format_number((confidence or 0) * 100, 1)}%)" if confidence is not None else "")
                        + "."
                    ),
                    (
                        f"توقع الهدف: {predicted_goal_label}"
                        + (f" (ثقة {_format_number((confidence or 0) * 100, 1)}%)" if confidence is not None else "")
                        + "."
                    ),
                    (
                        f"توقع الهدف: {predicted_goal_label}"
                        + (f" (ثقة {_format_number((confidence or 0) * 100, 1)}%)" if confidence is not None else "")
                        + "."
                    ),
                )
                reply_parts.append(goal_reply)
                payload["goal_prediction"] = result
                payload["goal_features_used"] = goal_features
            except FileNotFoundError:
                reply_parts.append(
                    _lang_reply(
                        language,
                        "Goal model is not available yet. Train `model_goal.pkl` first.",
                        "نموذج توقع الهدف غير متاح بعد. درّب `model_goal.pkl` أولًا.",
                        "نموذج توقع الهدف مش جاهز. درّب `model_goal.pkl` أول.",
                    )
                )

    if want_success:
        success_features, missing = _build_success_prediction_payload(profile, tracking_summary)
        if missing:
            reply_parts.append(_ml_missing_fields_reply(language, "success", missing))
        else:
            try:
                result = predict_success(success_features)
                prediction_flag = int(result.get("success_prediction", 0) or 0)
                probability = _to_float(result.get("success_probability"))
                status_text = _lang_reply(
                    language,
                    "likely on track" if prediction_flag == 1 else "at risk / needs adjustment",
                    "غالبًا على المسار الصحيح" if prediction_flag == 1 else "مُعرّض للتأخر ويحتاج تعديل",
                    "غالبًا ماشي صح" if prediction_flag == 1 else "في خطر تأخير وبدها تعديل",
                )
                success_reply = _lang_reply(
                    language,
                    (
                        "Success prediction: "
                        + (f"{_format_number((probability or 0) * 100, 1)}% " if probability is not None else "")
                        + f"({status_text})."
                    ),
                    (
                        "توقع النجاح: "
                        + (f"{_format_number((probability or 0) * 100, 1)}% " if probability is not None else "")
                        + f"({status_text})."
                    ),
                    (
                        "توقع النجاح: "
                        + (f"{_format_number((probability or 0) * 100, 1)}% " if probability is not None else "")
                        + f"({status_text})."
                    ),
                )
                reply_parts.append(success_reply)
                payload["success_prediction"] = result
                payload["success_features_used"] = success_features
            except FileNotFoundError:
                reply_parts.append(
                    _lang_reply(
                        language,
                        "Success model is not available yet. Train `model_success.pkl` first.",
                        "نموذج توقع النجاح غير متاح بعد. درّب `model_success.pkl` أولًا.",
                        "نموذج توقع النجاح مش جاهز. درّب `model_success.pkl` أول.",
                    )
                )

    if not reply_parts:
        return None

    return "\n".join(reply_parts), payload


def _status_label(language: str, status: str) -> str:
    status_key = status.strip().lower()
    if status_key == "ahead of schedule":
        return _lang_reply(language, "Ahead of schedule", "متقدم عن الخطة", "متقدّم عن الخطة")
    if status_key == "behind schedule":
        return _lang_reply(language, "Behind schedule", "متأخر عن الخطة", "متأخر عن الخطة")
    return _lang_reply(language, "On track", "على المسار الصحيح", "على المسار")


def _performance_missing_data_reply(language: str, missing_fields: list[str]) -> str:
    fields_text = ", ".join(missing_fields)
    quick_example = (
        "وزني الحالي 92، هدفي 85، تغير وزني الأسبوعي -0.5"
    )
    return _lang_reply(
        language,
        (
            "I can estimate how long is left, but I need a few missing details: "
            f"{fields_text}. "
            "Send them in plain text, for example: "
            f"{quick_example}"
        ),
        (
            "بقدر أحسب لك كم ضايل، بس ناقصني شوية بيانات: "
            f"{fields_text}. "
            "ابعثهم كتابة بشكل بسيط مثل: "
            f"{quick_example}"
        ),
        (
            "بقدر أحسب لك قديش ضايل، بس ناقصني بيانات: "
            f"{fields_text}. "
            "ابعتهم بشكل بسيط مثل: "
            f"{quick_example}"
        ),
    )


def _performance_analysis_reply(
    language: str,
    profile: dict[str, Any],
    tracking_summary: Optional[dict[str, Any]],
) -> str:
    if not isinstance(tracking_summary, dict):
        return _performance_missing_data_reply(
            language,
            ["goal.type", "goal.current_weight", "goal.target_weight", "weekly_stats.weight_change or weekly_stats.weight_change_history"],
        )

    goal_data = tracking_summary.get("goal") if isinstance(tracking_summary.get("goal"), dict) else {}
    weekly_stats = (
        tracking_summary.get("weekly_stats")
        if isinstance(tracking_summary.get("weekly_stats"), dict)
        else {}
    )
    monthly_stats = (
        tracking_summary.get("monthly_stats")
        if isinstance(tracking_summary.get("monthly_stats"), dict)
        else {}
    )

    goal_type_raw = _dict_get_any(goal_data, ["type", "goal_type"]) or profile.get("goal")
    goal_type = _normalize_goal(goal_type_raw)

    current_weight = _to_float(
        _dict_get_any(goal_data, ["current_weight", "currentWeight", "weight"]) or profile.get("weight")
    )
    target_weight = _to_float(_dict_get_any(goal_data, ["target_weight", "targetWeight"]))

    weekly_weight_change_point = _to_float(
        _dict_get_any(weekly_stats, ["weight_change", "weekly_weight_change", "weightChange"])
    )
    monthly_weight_change = _to_float(_dict_get_any(monthly_stats, ["weight_change", "monthly_weight_change"]))
    weight_change_series_all = _extract_weight_change_series(tracking_summary, weekly_stats)
    weight_change_series_recent = weight_change_series_all[-4:]
    weekly_weight_change = _average(weight_change_series_recent)
    if weekly_weight_change is None and weekly_weight_change_point is not None:
        weekly_weight_change = weekly_weight_change_point
    if weekly_weight_change is None and monthly_weight_change is not None:
        weekly_weight_change = monthly_weight_change / 4.0

    strength_increase_monthly = _to_float(
        _dict_get_any(monthly_stats, ["strength_increase_percent", "strength_increase_pct", "strength_percent"])
    )
    target_strength_increase = _to_float(
        _dict_get_any(goal_data, ["target_strength_increase_percent", "target_strength_percent"])
    )

    workout_days = _to_float(_dict_get_any(weekly_stats, ["workout_days"]))
    planned_days = _to_float(_dict_get_any(weekly_stats, ["planned_days"]))
    avg_calories = _to_float(_dict_get_any(weekly_stats, ["avg_calories", "average_calories"]))
    avg_protein = _to_float(_dict_get_any(weekly_stats, ["avg_protein", "average_protein"]))
    sleep_avg_hours = _to_float(_dict_get_any(weekly_stats, ["sleep_avg_hours", "sleep_hours"]))

    consistency_percent = _to_float(
        _dict_get_any(monthly_stats, ["consistency_percent", "consistency_pct"])
    )
    if consistency_percent is None:
        adherence_score = _to_float(_dict_get_any(tracking_summary, ["adherence_score"]))
        if adherence_score is not None:
            consistency_percent = adherence_score * 100.0

    trend_weeks_count = len(weight_change_series_recent)
    trend_series_text = ", ".join(f"{value:+.2f}" for value in weight_change_series_recent)
    trend_variability = _mean_abs_deviation(weight_change_series_recent)

    missing_fields: list[str] = []
    weight_goal_mode = goal_type == "fat_loss" or target_weight is not None

    if weight_goal_mode:
        if current_weight is None:
            missing_fields.append("goal.current_weight")
        if target_weight is None:
            missing_fields.append("goal.target_weight")
        if weekly_weight_change is None:
            missing_fields.append("weekly_stats.weight_change or weekly_stats.weight_change_history")
    elif goal_type == "muscle_gain":
        if strength_increase_monthly is None and weekly_weight_change is None:
            missing_fields.append("monthly_stats.strength_increase_percent or weekly_stats.weight_change/weight_change_history")
        if target_weight is None and target_strength_increase is None:
            missing_fields.append("goal.target_weight or goal.target_strength_increase_percent")
    else:
        if weekly_weight_change is None and strength_increase_monthly is None:
            missing_fields.append("weekly_stats.weight_change/weight_change_history or monthly_stats.strength_increase_percent")

    if missing_fields:
        return _performance_missing_data_reply(language, missing_fields)

    status = "on track"
    weeks_remaining: Optional[float] = None
    remaining_weight: Optional[float] = None

    if target_weight is not None and current_weight is not None and weekly_weight_change is not None:
        remaining_weight = target_weight - current_weight
        if abs(remaining_weight) < 0.05:
            status = "ahead of schedule"
            weeks_remaining = 0.0
        elif abs(weekly_weight_change) < 1e-9:
            status = "behind schedule"
        else:
            toward_target = weekly_weight_change * remaining_weight > 0
            if not toward_target:
                status = "behind schedule"
            else:
                weeks_remaining = abs(remaining_weight) / abs(weekly_weight_change)
                weekly_pct = abs(weekly_weight_change) / max(current_weight, 1e-6) * 100.0
                if goal_type == "fat_loss":
                    if weekly_pct > 1.0:
                        status = "ahead of schedule"
                    elif weekly_pct >= 0.25:
                        status = "on track"
                    else:
                        status = "behind schedule"
                elif goal_type == "muscle_gain":
                    if weekly_pct > 0.5:
                        status = "ahead of schedule"
                    elif weekly_pct >= 0.1:
                        status = "on track"
                    else:
                        status = "behind schedule"
                else:
                    status = "on track"

                if trend_weeks_count >= 2:
                    toward_weeks = sum(1 for change in weight_change_series_recent if (change * remaining_weight) > 0)
                    toward_ratio = toward_weeks / trend_weeks_count
                    if toward_ratio < 0.5:
                        status = "behind schedule"
                    elif toward_ratio < 0.75 and status == "ahead of schedule":
                        status = "on track"

                    if trend_variability is not None and abs(weekly_weight_change) > 1e-9:
                        variability_ratio = trend_variability / abs(weekly_weight_change)
                        if variability_ratio > 1.6:
                            status = "behind schedule"
                        elif variability_ratio > 1.1 and status == "ahead of schedule":
                            status = "on track"

    elif goal_type == "muscle_gain" and target_strength_increase is not None and strength_increase_monthly is not None:
        if strength_increase_monthly <= 0:
            status = "behind schedule"
        else:
            strength_remaining = max(0.0, target_strength_increase - strength_increase_monthly)
            weeks_remaining = (strength_remaining / strength_increase_monthly) * 4.0
            if strength_increase_monthly >= 5.0:
                status = "ahead of schedule"
            elif strength_increase_monthly > 0:
                status = "on track"
            else:
                status = "behind schedule"

    if consistency_percent is not None and consistency_percent < 70.0:
        status = "behind schedule"

    status_text = _status_label(language, status)
    weeks_text = "N/A" if weeks_remaining is None else f"{weeks_remaining:.1f}"

    workout_adherence_line = "N/A"
    if workout_days is not None and planned_days is not None and planned_days > 0:
        workout_adherence_line = f"{(workout_days / planned_days) * 100:.0f}% ({int(workout_days)}/{int(planned_days)} days)"

    calorie_target = _to_float(_dict_get_any(weekly_stats, ["target_calories"])) or _to_float(profile.get("target_calories"))
    calorie_delta: Optional[float] = None
    if avg_calories is not None and calorie_target is not None:
        calorie_delta = avg_calories - calorie_target

    recommendations: list[str] = []
    if goal_type == "fat_loss":
        if calorie_delta is not None and calorie_delta > 0:
            recommendations.append(f"Calories: reduce daily intake by ~{int(min(300, max(120, calorie_delta)))} kcal to match deficit target.")
        elif status == "ahead of schedule":
            recommendations.append("Calories: fat loss speed is high; add 100-150 kcal/day to protect recovery and muscle.")
        else:
            recommendations.append("Training volume: keep 10-16 hard sets per major muscle/week; add +2 sets for weak muscles if needed.")
    elif goal_type == "muscle_gain":
        if status == "behind schedule":
            recommendations.append("Volume: increase by +2 to +4 hard sets per target muscle/week and track progressive overload.")
        else:
            recommendations.append("Volume: keep current progression; maintain controlled overload weekly.")
        if calorie_delta is not None and calorie_delta < 0:
            recommendations.append(f"Calories: add ~{int(min(300, max(120, abs(calorie_delta))))} kcal/day to support muscle gain.")
    else:
        recommendations.append("Volume: adjust weekly load by +/-10% based on fatigue and performance trend.")

    if avg_protein is not None and current_weight is not None:
        protein_per_kg = avg_protein / max(current_weight, 1e-6)
        if protein_per_kg < 1.6:
            recommendations.append("Protein: increase toward 1.6-2.2 g/kg/day for better adaptation.")
    if sleep_avg_hours is not None and sleep_avg_hours < 7.0:
        recommendations.append("Recovery: increase sleep to 7-9 h/night to improve strength and body-composition progress.")

    if not recommendations:
        recommendations.append("Keep consistency high and review weekly data before adjusting plan variables.")

    recommendations_block = "\n".join(f"{idx}. {text}" for idx, text in enumerate(recommendations[:3], start=1))

    if trend_weeks_count >= 2:
        rate_line_en = f"Rate of progress (trend last {trend_weeks_count} weeks): {_format_number(weekly_weight_change)} kg/week"
        rate_line_ar_fusha = f"معدل التقدم (اتجاه آخر {trend_weeks_count} أسابيع): {_format_number(weekly_weight_change)} كغ/أسبوع"
        rate_line_ar_jordanian = f"معدل التقدم (اتجاه آخر {trend_weeks_count} أسابيع): {_format_number(weekly_weight_change)} كيلو/أسبوع"
        trend_details_en = f"Recent weekly changes: {trend_series_text} kg/week\n"
        trend_details_ar_fusha = f"تغيرات الأسابيع الأخيرة: {trend_series_text} كغ/أسبوع\n"
        trend_details_ar_jordanian = f"تغيرات آخر الأسابيع: {trend_series_text} كيلو/أسبوع\n"
    else:
        rate_line_en = f"Rate of progress: {_format_number(weekly_weight_change)} kg/week"
        rate_line_ar_fusha = f"معدل التقدم: {_format_number(weekly_weight_change)} كغ/أسبوع"
        rate_line_ar_jordanian = f"معدل التقدم: {_format_number(weekly_weight_change)} كيلو/أسبوع"
        trend_details_en = ""
        trend_details_ar_fusha = ""
        trend_details_ar_jordanian = ""

    if trend_weeks_count == 0:
        if weekly_weight_change_point is not None:
            trend_details_en = "Rate source: single weekly point.\n"
            trend_details_ar_fusha = "مصدر المعدل: نقطة أسبوعية واحدة.\n"
            trend_details_ar_jordanian = "مصدر المعدل: نقطة أسبوعية وحدة.\n"
        elif monthly_weight_change is not None:
            trend_details_en = "Rate source: monthly change divided by 4.\n"
            trend_details_ar_fusha = "مصدر المعدل: التغير الشهري مقسوم على 4.\n"
            trend_details_ar_jordanian = "مصدر المعدل: التغير الشهري مقسوم على 4.\n"

    return _lang_reply(
        language,
        (
            f"Status: {status_text}\n"
            + rate_line_en
            + (f" | Strength: {_format_number(strength_increase_monthly)}%/month" if strength_increase_monthly is not None else "")
            + "\n"
            + trend_details_en
            + (
                f"Remaining weight difference: {_format_number(remaining_weight)} kg\n"
                if remaining_weight is not None
                else ""
            )
            + f"Estimated time to target: {weeks_text} weeks\n"
            + f"Consistency: {_format_number(consistency_percent, 1)}% | Workout adherence: {workout_adherence_line}\n"
            + (
                f"Calories: avg {_format_number(avg_calories, 0)} kcal"
                + (f" vs target {_format_number(calorie_target, 0)} ({_format_number(calorie_delta, 0)} delta)" if calorie_target is not None and calorie_delta is not None else "")
                + "\n"
                if avg_calories is not None
                else ""
            )
            + "Recommendations:\n"
            + recommendations_block
        ),
        (
            f"الحالة: {status_text}\n"
            + rate_line_ar_fusha
            + (f" | القوة: {_format_number(strength_increase_monthly)}%/شهر" if strength_increase_monthly is not None else "")
            + "\n"
            + trend_details_ar_fusha
            + (
                f"فرق الوزن المتبقي: {_format_number(remaining_weight)} كغ\n"
                if remaining_weight is not None
                else ""
            )
            + f"الوقت المتوقع للوصول للهدف: {weeks_text} أسبوع\n"
            + f"نسبة الالتزام: {_format_number(consistency_percent, 1)}% | التزام التمرين: {workout_adherence_line}\n"
            + (
                f"السعرات: متوسط {_format_number(avg_calories, 0)} سعرة"
                + (f" مقابل الهدف {_format_number(calorie_target, 0)} (فرق {_format_number(calorie_delta, 0)})" if calorie_target is not None and calorie_delta is not None else "")
                + "\n"
                if avg_calories is not None
                else ""
            )
            + "التوصيات:\n"
            + recommendations_block
        ),
        (
            f"الحالة: {status_text}\n"
            + rate_line_ar_jordanian
            + (f" | القوة: {_format_number(strength_increase_monthly)}%/شهر" if strength_increase_monthly is not None else "")
            + "\n"
            + trend_details_ar_jordanian
            + (
                f"فرق الوزن المتبقي: {_format_number(remaining_weight)} كيلو\n"
                if remaining_weight is not None
                else ""
            )
            + f"الوقت المتوقع توصل للهدف: {weeks_text} أسبوع\n"
            + f"الالتزام: {_format_number(consistency_percent, 1)}% | التزام التمرين: {workout_adherence_line}\n"
            + (
                f"السعرات: متوسط {_format_number(avg_calories, 0)}"
                + (f" مقابل الهدف {_format_number(calorie_target, 0)} (فرق {_format_number(calorie_delta, 0)})" if calorie_target is not None and calorie_delta is not None else "")
                + "\n"
                if avg_calories is not None
                else ""
            )
            + "التوصيات:\n"
            + recommendations_block
        ),
    )


def _general_llm_reply(
    user_message: str,
    language: str,
    profile: dict[str, Any],
    tracking_summary: Optional[dict[str, Any]],
    memory: MemorySystem,
    state: Optional[dict[str, Any]] = None,
    recent_messages: Optional[list[dict[str, Any]]] = None,
) -> str:
    language_instructions = {
        "en": "Reply in clear English.",
        "ar_fusha": "رد باللغة العربية الفصحى.",
        "ar_jordanian": "احكِ باللهجة الأردنية بشكل واضح.",
    }.get(language, "Reply in English.")

    display_name = _profile_display_name(profile)
    state = state or {}
    plan_snapshot = state.get("plan_snapshot", {})
    nutrition_kb_context = _nutrition_kb_context(user_message, profile, top_k=3)

    system_prompt = (
        "You are a professional AI fitness coach.\n"
        "You ONLY answer fitness, training, sports performance, and nutrition topics.\n"
        "If user asks outside this domain, politely refuse and redirect back to fitness.\n"
        "Be warm and supportive, but practical.\n"
        "Personalize responses using user profile fields (name, goal, age, height, weight, health constraints).\n"
        "For weekly/monthly performance questions, be analytical and numeric.\n"
        "Compare recent data against the goal, calculate the rate of progress, classify status (On track / Ahead / Behind), and estimate weeks remaining when data is sufficient.\n"
        "Never guess missing metrics; explicitly ask for the exact missing fields.\n"
        "When nutrition knowledge snippets are provided in context, prioritize them over generic advice.\n"
        "If progress is weak or user reports no body change, ask about sleep, hydration, meal adherence, and workout execution before giving final advice.\n"
        "When user asks about exercises, guide them and mention they can use /workouts for muscle-specific exercise explorer.\n"
        "Keep responses concise but useful.\n"
        f"{language_instructions}\n"
    )

    context_lines = [
        f"User name: {display_name or 'Unknown'}",
        f"User profile: {profile}",
        f"Tracking summary: {tracking_summary or {}}",
        f"Plan snapshot: {plan_snapshot or {}}",
        f"Plans recently deleted flag: {bool(state.get('plans_recently_deleted', False))}",
    ]
    if nutrition_kb_context:
        context_lines.append("Nutrition reference snippets (from DATAFORPROJECT.pdf):")
        context_lines.append(nutrition_kb_context)
    messages = [{"role": "system", "content": system_prompt + '\n'.join(context_lines)}]

    external_history = _normalize_recent_messages(recent_messages)
    if external_history:
        messages.extend(external_history[-10:])
    else:
        messages.extend(memory.get_conversation_history()[-8:])

    last_history_text = normalize_text(messages[-1]["content"]) if len(messages) > 1 else ""
    if last_history_text != normalize_text(user_message):
        messages.append({"role": "user", "content": user_message})
    return LLM.chat_completion(messages, max_tokens=500)


@app.get("/health")
def health() -> dict[str, Any]:
    dataset_summary = DATASET_REGISTRY.summary()
    return {
        "status": "ok",
        "provider": LLM.active_provider,
        "model": LLM.active_model,
        "chat_response_mode": CHAT_RESPONSE_MODE,
        "response_dataset_source": str(RESPONSE_DATASET_DIR),
        "nutrition_knowledge_loaded": NUTRITION_KB.ready,
        "nutrition_knowledge_source": str(NUTRITION_KB.data_path),
        "dataset_registry_files": dataset_summary.get("files_count", 0),
        "dataset_registry_generated_at": dataset_summary.get("generated_at"),
        "features": [
            "domain_router",
            "moderation",
            "memory",
            "workout_plans",
            "nutrition_plans",
            "nutrition_knowledge",
            "plan_approval",
            "plan_options",
            "multilingual",
            "tracking_data_extraction",
            "deterministic_performance_analysis",
            "four_week_trend_scoring",
            "ml_goal_prediction",
            "ml_success_prediction",
            "ml_plan_intent_prediction",
            "logic_engine_metrics",
            "dataset_registry_all_files",
        ],
    }


@app.get("/datasets/summary")
def datasets_summary() -> dict[str, Any]:
    return {"status": "ok", "summary": DATASET_REGISTRY.summary()}


@app.get("/datasets/search")
def datasets_search(q: str = Query(..., min_length=1), top_k: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
    results = DATASET_REGISTRY.search(q, top_k=top_k)
    return {"status": "ok", "query": q, "count": len(results), "results": results}


@app.get("/datasets/tag/{tag}")
def datasets_by_tag(tag: str) -> dict[str, Any]:
    items = DATASET_REGISTRY.tagged_files(tag)
    slim = [
        {
            "relative_path": item.get("relative_path"),
            "category": item.get("category"),
            "extension": item.get("extension"),
            "size_bytes": item.get("size_bytes"),
            "tags": item.get("tags", []),
        }
        for item in items
    ]
    return {"status": "ok", "tag": tag, "count": len(slim), "files": slim}


@app.post("/ml/predict-goal")
def ml_predict_goal(req: GoalPredictionRequest) -> dict[str, Any]:
    try:
        payload = req.model_dump()
        result = predict_goal(payload)
        return {"status": "ok", "prediction": result}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Goal model unavailable: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Goal prediction failed: {exc}") from exc


@app.post("/ml/predict-success")
def ml_predict_success(req: SuccessPredictionRequest) -> dict[str, Any]:
    try:
        payload = req.model_dump()
        result = predict_success(payload)
        return {"status": "ok", "prediction": result}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Success model unavailable: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Success prediction failed: {exc}") from exc


@app.post("/ml/predict-plan-intent")
def ml_predict_plan_intent(req: PlanIntentPredictionRequest) -> dict[str, Any]:
    try:
        result = predict_plan_intent(req.message)
        return {"status": "ok", "prediction": result}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Plan-intent model unavailable: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Plan-intent prediction failed: {exc}") from exc


@app.post("/logic/evaluate")
def logic_evaluate(req: LogicEvaluationRequest) -> dict[str, Any]:
    try:
        metrics = evaluate_logic_metrics(
            start_value=req.start_value,
            current_value=req.current_value,
            target_value=req.target_value,
            direction=req.direction,
            weight_history=req.weight_history,
            previous_value=req.previous_value,
            elapsed_weeks=req.elapsed_weeks,
        )
        return {"status": "ok", "metrics": metrics.__dict__}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Logic evaluation failed: {exc}") from exc


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    user_id = _normalize_user_id(req.user_id)
    conversation_id = _normalize_conversation_id(req.conversation_id, user_id)
    state = _get_user_state(user_id)
    profile = _build_profile(req, state)
    language = _detect_language(req.language or "en", req.message, profile)
    recent_messages = _normalize_recent_messages(req.recent_messages)

    _persist_profile_context(profile, state)
    if req.tracking_summary:
        state["last_progress_summary"] = _merge_tracking_summaries(
            state.get("last_progress_summary"),
            req.tracking_summary,
        )
    _update_plan_snapshot_state(state, req.plan_snapshot)
    tracking_summary = state.get("last_progress_summary")

    user_input = _repair_mojibake(req.message.strip())
    if not user_input:
        return ChatResponse(
            reply="Please send a valid message." if language == "en" else "أرسل رسالة واضحة.",
            conversation_id=conversation_id,
            language=language,
        )

    message_tracking_summary = _extract_tracking_summary_from_message(user_input, profile)
    if message_tracking_summary:
        tracking_summary = _merge_tracking_summaries(tracking_summary, message_tracking_summary)
        state["last_progress_summary"] = tracking_summary

    memory = _get_memory_session(user_id, conversation_id)
    memory.add_user_message(user_input)

    _, has_bad_words = MODERATION.filter_content(user_input, language=language)
    if has_bad_words:
        fallback = MODERATION.get_safe_fallback(language)
        memory.add_assistant_message(fallback)
        return ChatResponse(reply=fallback, conversation_id=conversation_id, language=language)

    lowered = normalize_text(user_input)

    pending_options_payload = state.get("pending_plan_options")
    if pending_options_payload:
        pending_conv = pending_options_payload.get("conversation_id")
        if pending_conv and pending_conv != conversation_id:
            state["pending_plan_options"] = None
            pending_options_payload = None
    if pending_options_payload:
        pending_options = pending_options_payload.get("options", [])
        pending_options_type = str(pending_options_payload.get("plan_type", "workout"))
        selected_idx = _extract_plan_choice_index(user_input, len(pending_options))

        if selected_idx is not None:
            selected_plan = deepcopy(pending_options[selected_idx])
            plan_id = selected_plan["id"]
            PENDING_PLANS[plan_id] = {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "plan_type": pending_options_type,
                "plan": selected_plan,
                "approved": False,
                "created_at": datetime.utcnow().isoformat(),
            }
            state["last_pending_plan_id"] = plan_id
            state["pending_plan_options"] = None
            state["pending_plan_type"] = None

            reply = _format_plan_preview(pending_options_type, selected_plan, language)
            memory.add_assistant_message(reply)
            return ChatResponse(
                reply=reply,
                conversation_id=conversation_id,
                language=language,
                action="ask_plan",
                data={"plan_id": plan_id, "plan_type": pending_options_type, "plan": selected_plan},
            )

        if _contains_any(user_input, PLAN_REFRESH_KEYWORDS):
            profile = _build_profile(req, state)
            if pending_options_type == "nutrition":
                refreshed_options = _generate_nutrition_plan_options(profile, language, count=5)
            else:
                refreshed_options = _generate_workout_plan_options(profile, language, count=5)
            state["pending_plan_options"] = {
                "plan_type": pending_options_type,
                "options": refreshed_options,
                "conversation_id": conversation_id,
            }
            reply = _format_plan_options_preview(pending_options_type, refreshed_options, language)
            memory.add_assistant_message(reply)
            return ChatResponse(
                reply=reply,
                conversation_id=conversation_id,
                language=language,
                action="choose_plan",
                data={"plan_type": pending_options_type, "options_count": len(refreshed_options)},
            )

        reply = _format_plan_options_preview(pending_options_type, pending_options, language)
        memory.add_assistant_message(reply)
        return ChatResponse(
            reply=reply,
            conversation_id=conversation_id,
            language=language,
            action="choose_plan",
            data={"plan_type": pending_options_type, "options_count": len(pending_options)},
        )

    latest_plan_id = state.get("last_pending_plan_id")
    if latest_plan_id and latest_plan_id in PENDING_PLANS:
        if _contains_any(lowered, APPROVE_KEYWORDS):
            plan_record = PENDING_PLANS[latest_plan_id]
            plan_record["approved"] = True
            state["last_pending_plan_id"] = None
            if language == "en":
                reply = "Plan approved. I added it for your schedule flow. You can track completion daily."
            elif language == "ar_fusha":
                reply = "تم اعتماد الخطة. يمكنك الآن متابعتها يوميًا داخل صفحة الجدول."
            else:
                reply = "تم اعتماد الخطة. هلا بتقدر تتابعها يوم بيوم بصفحة الجدول."
            memory.add_assistant_message(reply)
            return ChatResponse(
                reply=reply,
                conversation_id=conversation_id,
                language=language,
                action="plan_approved",
                data={
                    "plan_id": latest_plan_id,
                    "approved_plan": {
                        "type": plan_record["plan_type"],
                        "plan": plan_record["plan"],
                    },
                },
            )
        if _contains_any(lowered, REJECT_KEYWORDS):
            state["last_pending_plan_id"] = None
            if language == "en":
                reply = "No problem. I canceled this draft. Tell me what to change and I will regenerate it."
            elif language == "ar_fusha":
                reply = "لا مشكلة. ألغيت هذه المسودة. أخبرني ما الذي تريد تغييره وسأعيد التوليد."
            else:
                reply = "تمام، لغيت المسودة. احكيلي شو بدك أغير وبرجع ببنيها."
            memory.add_assistant_message(reply)
            return ChatResponse(
                reply=reply,
                conversation_id=conversation_id,
                language=language,
                action="plan_rejected",
                data={"plan_id": latest_plan_id},
            )

    # Handle numeric progress/performance analysis before strict dataset fallback.
    # This keeps the analytical path reachable even when strict intent matching is enabled.
    if _is_performance_analysis_request(user_input, message_tracking_summary):
        performance_reply = _performance_analysis_reply(language, profile, tracking_summary)
        memory.add_assistant_message(performance_reply)
        return ChatResponse(reply=performance_reply, conversation_id=conversation_id, language=language)

    # Strict dataset mode:
    # - Chat replies are sourced only from conversation_intents.json.
    # - Plan options are sourced only from workout_programs.json / nutrition_programs.json.
    # - Legacy non-dataset flows are disabled.
    state["pending_field"] = None
    state["pending_plan_type"] = None
    state["pending_diagnostic"] = None
    state["pending_diagnostic_conversation_id"] = None

    requested_plan_type, plan_intent_meta = _resolve_plan_type_from_message(user_input)
    if requested_plan_type in {"workout", "nutrition"}:
        inferred_goal, inferred_confidence, inferred_by_ml = _infer_goal_for_plan(profile, tracking_summary)
        plan_profile = dict(profile)
        plan_profile["goal"] = inferred_goal

        if requested_plan_type == "workout":
            options = _generate_workout_plan_options(plan_profile, language, count=5)
        else:
            options = _generate_nutrition_plan_options(plan_profile, language, count=5)

        if not options:
            reply = _dataset_intent_response("out_of_scope", language, seed=user_input) or _dataset_fallback_reply(
                language, seed=user_input
            )
            memory.add_assistant_message(reply)
            return ChatResponse(reply=reply, conversation_id=conversation_id, language=language)

        state["pending_plan_options"] = {
            "plan_type": requested_plan_type,
            "options": options,
            "conversation_id": conversation_id,
        }
        if inferred_by_ml:
            state["inferred_goal"] = inferred_goal

        reply = _format_plan_options_preview(requested_plan_type, options, language)

        info_lines: list[str] = []
        if inferred_by_ml:
            goal_label = _profile_goal_label(inferred_goal, language)
            conf_text = (
                f" ({_format_number((inferred_confidence or 0.0) * 100, 1)}%)"
                if inferred_confidence is not None
                else ""
            )
            info_lines.append(
                _lang_reply(
                    language,
                    f"Auto-inferred goal from training data: {goal_label}{conf_text}.",
                    f"تم استنتاج الهدف تلقائيًا من بيانات التدريب: {goal_label}{conf_text}.",
                    f"استنتجت هدفك تلقائيًا من بيانات التدريب: {goal_label}{conf_text}.",
                )
            )

        if plan_intent_meta:
            predicted_intent = str(plan_intent_meta.get("predicted_intent", requested_plan_type))
            intent_confidence = _to_float(plan_intent_meta.get("confidence"))
            conf_text = (
                f" ({_format_number((intent_confidence or 0.0) * 100, 1)}%)"
                if intent_confidence is not None
                else ""
            )
            if _is_generic_plan_request(user_input):
                info_lines.append(
                    _lang_reply(
                        language,
                        f"Detected plan type automatically: {predicted_intent}{conf_text}.",
                        f"تم تحديد نوع الخطة تلقائيًا: {predicted_intent}{conf_text}.",
                        f"حددّت نوع الخطة تلقائيًا: {predicted_intent}{conf_text}.",
                    )
                )

        if info_lines:
            reply = "\n".join(info_lines + [reply])

        memory.add_assistant_message(reply)
        return ChatResponse(
            reply=reply,
            conversation_id=conversation_id,
            language=language,
            action="choose_plan",
            data={
                "plan_type": requested_plan_type,
                "options_count": len(options),
                "inferred_goal": inferred_goal,
                "inferred_goal_confidence": inferred_confidence,
                "plan_intent_prediction": plan_intent_meta or {},
            },
        )

    ml_prediction_payload = _ml_prediction_chat_response(user_input, language, profile, tracking_summary)
    if ml_prediction_payload:
        ml_reply, ml_data = ml_prediction_payload
        state["last_ml_prediction"] = ml_data
        memory.add_assistant_message(ml_reply)
        return ChatResponse(
            reply=ml_reply,
            conversation_id=conversation_id,
            language=language,
            action="ml_prediction",
            data=ml_data,
        )

    # Always give priority to deterministic dataset replies before any routing/LLM work.
    # This fixes cases where intents were defined in conversation_intents.json but never surfaced.
    dataset_reply = _dataset_conversation_reply(user_input, language)
    if dataset_reply:
        memory.add_assistant_message(dataset_reply)
        return ChatResponse(reply=dataset_reply, conversation_id=conversation_id, language=language)

    if CHAT_RESPONSE_MODE != "dataset_only":
        in_domain, _score = ROUTER.is_in_domain(user_input, language=language)
        if (not in_domain) and _contains_any(user_input, STRONG_DOMAIN_KEYWORDS):
            in_domain = True
        if not in_domain:
            out_reply = _strict_out_of_scope_reply(language)
            memory.add_assistant_message(out_reply)
            return ChatResponse(reply=out_reply, conversation_id=conversation_id, language=language)

        # Keep deterministic short conversational replies for very short inputs.
        dataset_reply = _dataset_conversation_reply(user_input, language)
        if dataset_reply and len(normalize_text(user_input).split()) <= 4:
            memory.add_assistant_message(dataset_reply)
            return ChatResponse(reply=dataset_reply, conversation_id=conversation_id, language=language)

        llm_reply = _general_llm_reply(
            user_message=user_input,
            language=language,
            profile=profile,
            tracking_summary=tracking_summary,
            memory=memory,
            state=state,
            recent_messages=recent_messages,
        )
        if llm_reply.startswith("Ollama error:") or llm_reply.startswith("Ollama is not reachable"):
            llm_reply = _lang_reply(
                language,
                "Local AI is unavailable. This project runs free with Ollama. Start Ollama, run `ollama pull llama3.2:3b`, then retry.",
                "الذكاء المحلي غير متاح حالياً. هذا المشروع مجاني عبر Ollama. شغّل Ollama ثم نفّذ `ollama pull llama3.2:3b` وبعدها أعد المحاولة.",
                "الذكاء المحلي واقف حالياً. المشروع مجاني على Ollama. شغّل Ollama واعمل `ollama pull llama3.2:3b` وجرّب مرة ثانية.",
            )

        filtered_reply, _ = MODERATION.filter_content(llm_reply, language=language)
        memory.add_assistant_message(filtered_reply)
        return ChatResponse(reply=filtered_reply, conversation_id=conversation_id, language=language)

    dataset_reply = _dataset_conversation_reply(user_input, language)
    if dataset_reply:
        memory.add_assistant_message(dataset_reply)
        return ChatResponse(reply=dataset_reply, conversation_id=conversation_id, language=language)

    out_reply = _dataset_intent_response("out_of_scope", language, seed=user_input) or _dataset_fallback_reply(
        language, seed=user_input
    )
    memory.add_assistant_message(out_reply)
    return ChatResponse(reply=out_reply, conversation_id=conversation_id, language=language)

    pending_field = state.get("pending_field")
    if pending_field:
        if _apply_profile_answer(pending_field, user_input, state):
            state["pending_field"] = None
            pending_plan_type = state.get("pending_plan_type")
            profile = _build_profile(req, state)
            if pending_plan_type:
                missing = _missing_fields_for_plan(pending_plan_type, profile)
                if missing:
                    state["pending_field"] = missing[0]
                    question = _missing_field_question(missing[0], language)
                    memory.add_assistant_message(question)
                    return ChatResponse(
                        reply=question,
                        conversation_id=conversation_id,
                        language=language,
                        action="ask_profile",
                        data={"missing_field": missing[0], "plan_type": pending_plan_type},
                    )
                if pending_plan_type == "workout":
                    options = _generate_workout_plan_options(profile, language, count=5)
                else:
                    options = _generate_nutrition_plan_options(profile, language, count=5)

                state["pending_plan_options"] = {"plan_type": pending_plan_type, "options": options}
                state["pending_plan_type"] = None
                reply = _format_plan_options_preview(pending_plan_type, options, language)
                memory.add_assistant_message(reply)
                return ChatResponse(
                    reply=reply,
                    conversation_id=conversation_id,
                    language=language,
                    action="choose_plan",
                    data={"plan_type": pending_plan_type, "options_count": len(options)},
                )
        else:
            question = _missing_field_question(pending_field, language)
            memory.add_assistant_message(question)
            return ChatResponse(
                reply=question,
                conversation_id=conversation_id,
                language=language,
                action="ask_profile",
                data={"missing_field": pending_field, "plan_type": state.get("pending_plan_type")},
            )

    pending_diagnostic = state.get("pending_diagnostic")
    pending_diagnostic_conversation_id = state.get("pending_diagnostic_conversation_id")
    if pending_diagnostic and pending_diagnostic_conversation_id and pending_diagnostic_conversation_id != conversation_id:
        pending_diagnostic = None
    if pending_diagnostic and not _contains_any(lowered, PROGRESS_CONCERN_KEYWORDS | TROUBLESHOOT_KEYWORDS):
        diag_in_domain, _ = ROUTER.is_in_domain(user_input, language=language)
        if not diag_in_domain:
            state["pending_diagnostic"] = None
            state["pending_diagnostic_conversation_id"] = None
            out_reply = _dataset_intent_response("out_of_scope", language, seed=user_input) or _dataset_fallback_reply(
                language, seed=user_input
            )
            memory.add_assistant_message(out_reply)
            return ChatResponse(reply=out_reply, conversation_id=conversation_id, language=language)

        if pending_diagnostic == "progress":
            prompt = (
                "The user answered my progress-diagnostic questions. "
                "Analyze likely bottlenecks (sleep, hydration, nutrition adherence, execution) "
                "and give a concrete fix for the next 7 days."
            )
        else:
            prompt = (
                "The user answered my exercise-diagnostic questions. "
                "Identify likely form/load issue, provide corrective cues, safer load adjustment, "
                "and when to stop and seek in-person assessment."
            )
        diagnostic_reply = _general_llm_reply(
            user_message=f"{prompt}\n\nUser answer: {user_input}",
            language=language,
            profile=profile,
            tracking_summary=tracking_summary,
            memory=memory,
            state=state,
            recent_messages=recent_messages,
        )
        state["pending_diagnostic"] = None
        state["pending_diagnostic_conversation_id"] = None
        filtered_diagnostic, _ = MODERATION.filter_content(diagnostic_reply, language=language)
        memory.add_assistant_message(filtered_diagnostic)
        return ChatResponse(reply=filtered_diagnostic, conversation_id=conversation_id, language=language)

    # Strict dataset mode:
    # - Conversational replies must come from conversation_intents.json
    # - Plan content must come from workout_programs.json / nutrition_programs.json
    # - Any unmatched general message gets out_of_scope from the dataset.
    is_plan_request = _is_workout_plan_request(user_input) or _is_nutrition_plan_request(user_input)
    if not is_plan_request:
        dataset_reply = _dataset_conversation_reply(user_input, language)
        if dataset_reply:
            memory.add_assistant_message(dataset_reply)
            return ChatResponse(reply=dataset_reply, conversation_id=conversation_id, language=language)

        out_reply = _dataset_intent_response("out_of_scope", language, seed=user_input) or _dataset_fallback_reply(
            language, seed=user_input
        )
        memory.add_assistant_message(out_reply)
        return ChatResponse(reply=out_reply, conversation_id=conversation_id, language=language)

    if _is_greeting_query(user_input):
        reply = _greeting_reply(language, profile)
        memory.add_assistant_message(reply)
        return ChatResponse(reply=reply, conversation_id=conversation_id, language=language)

    if _is_name_query(user_input):
        reply = _name_reply(language)
        memory.add_assistant_message(reply)
        return ChatResponse(reply=reply, conversation_id=conversation_id, language=language)

    if _is_how_are_you_query(user_input):
        reply = _how_are_you_reply(language)
        memory.add_assistant_message(reply)
        return ChatResponse(reply=reply, conversation_id=conversation_id, language=language)

    latest_plan_id = state.get("last_pending_plan_id")
    if latest_plan_id and latest_plan_id in PENDING_PLANS:
        if _contains_any(lowered, APPROVE_KEYWORDS):
            plan_record = PENDING_PLANS[latest_plan_id]
            plan_record["approved"] = True
            state["last_pending_plan_id"] = None
            if language == "en":
                reply = "Plan approved. I added it for your schedule flow. You can track completion daily."
            elif language == "ar_fusha":
                reply = "تم اعتماد الخطة. يمكنك الآن متابعتها يوميًا داخل صفحة الجدول."
            else:
                reply = "تم اعتماد الخطة. هلا بتقدر تتابعها يوم بيوم بصفحة الجدول."
            memory.add_assistant_message(reply)
            return ChatResponse(
                reply=reply,
                conversation_id=conversation_id,
                language=language,
                action="plan_approved",
                data={
                    "plan_id": latest_plan_id,
                    "approved_plan": {
                        "type": plan_record["plan_type"],
                        "plan": plan_record["plan"],
                    },
                },
            )
        if _contains_any(lowered, REJECT_KEYWORDS):
            state["last_pending_plan_id"] = None
            if language == "en":
                reply = "No problem. I canceled this draft. Tell me what to change and I will regenerate it."
            elif language == "ar_fusha":
                reply = "لا مشكلة. ألغيت هذه المسودة. أخبرني ما الذي تريد تغييره وسأعيد التوليد."
            else:
                reply = "تمام، لغيت المسودة. احكيلي شو بدك أغير وبرجع ببنيها."
            memory.add_assistant_message(reply)
            return ChatResponse(
                reply=reply,
                conversation_id=conversation_id,
                language=language,
                action="plan_rejected",
                data={"plan_id": latest_plan_id},
            )

    social_reply = _social_reply(user_input, language, profile)
    if social_reply:
        memory.add_assistant_message(social_reply)
        return ChatResponse(reply=social_reply, conversation_id=conversation_id, language=language)

    profile_reply = _profile_query_reply(user_input, language, profile, tracking_summary)
    if profile_reply:
        memory.add_assistant_message(profile_reply)
        return ChatResponse(reply=profile_reply, conversation_id=conversation_id, language=language)

    if _contains_any(lowered, PLAN_STATUS_KEYWORDS):
        status_reply = _plan_status_reply(language, state.get("plan_snapshot"))
        memory.add_assistant_message(status_reply)
        return ChatResponse(reply=status_reply, conversation_id=conversation_id, language=language)

    if _is_performance_analysis_request(user_input, message_tracking_summary):
        performance_reply = _performance_analysis_reply(language, profile, tracking_summary)
        memory.add_assistant_message(performance_reply)
        return ChatResponse(reply=performance_reply, conversation_id=conversation_id, language=language)

    if _contains_any(lowered, PROGRESS_CONCERN_KEYWORDS):
        state["pending_diagnostic"] = "progress"
        state["pending_diagnostic_conversation_id"] = conversation_id
        response = _progress_diagnostic_reply(language, profile, tracking_summary)
        memory.add_assistant_message(response)
        return ChatResponse(reply=response, conversation_id=conversation_id, language=language)

    if _contains_any(lowered, TROUBLESHOOT_KEYWORDS):
        state["pending_diagnostic"] = "exercise"
        state["pending_diagnostic_conversation_id"] = conversation_id
        response = _exercise_diagnostic_reply(language)
        memory.add_assistant_message(response)
        return ChatResponse(reply=response, conversation_id=conversation_id, language=language)

    in_domain, _score = ROUTER.is_in_domain(user_input, language=language)
    if not in_domain:
        out_reply = _dataset_intent_response("out_of_scope", language, seed=user_input) or _dataset_fallback_reply(
            language, seed=user_input
        )
        memory.add_assistant_message(out_reply)
        return ChatResponse(reply=out_reply, conversation_id=conversation_id, language=language)

    if _is_workout_plan_request(user_input):
        state["pending_plan_type"] = "workout"
        profile = _build_profile(req, state)
        missing = _missing_fields_for_plan("workout", profile)
        if missing:
            state["pending_field"] = missing[0]
            question = _missing_field_question(missing[0], language)
            memory.add_assistant_message(question)
            return ChatResponse(
                reply=question,
                conversation_id=conversation_id,
                language=language,
                action="ask_profile",
                data={"missing_field": missing[0], "plan_type": "workout"},
            )

        options = _generate_workout_plan_options(profile, language, count=5)
        state["pending_plan_options"] = {"plan_type": "workout", "options": options}
        state["pending_plan_type"] = None
        reply = _format_plan_options_preview("workout", options, language)
        memory.add_assistant_message(reply)
        return ChatResponse(
            reply=reply,
            conversation_id=conversation_id,
            language=language,
            action="choose_plan",
            data={"plan_type": "workout", "options_count": len(options)},
        )

    if _is_nutrition_plan_request(user_input):
        state["pending_plan_type"] = "nutrition"
        profile = _build_profile(req, state)
        missing = _missing_fields_for_plan("nutrition", profile)
        if missing:
            state["pending_field"] = missing[0]
            question = _missing_field_question(missing[0], language)
            memory.add_assistant_message(question)
            return ChatResponse(
                reply=question,
                conversation_id=conversation_id,
                language=language,
                action="ask_profile",
                data={"missing_field": missing[0], "plan_type": "nutrition"},
            )

        options = _generate_nutrition_plan_options(profile, language, count=5)
        state["pending_plan_options"] = {"plan_type": "nutrition", "options": options}
        state["pending_plan_type"] = None
        reply = _format_plan_options_preview("nutrition", options, language)
        memory.add_assistant_message(reply)
        return ChatResponse(
            reply=reply,
            conversation_id=conversation_id,
            language=language,
            action="choose_plan",
            data={"plan_type": "nutrition", "options_count": len(options)},
        )

    if _contains_any(lowered, PROGRESS_KEYWORDS):
        reply = _tracking_reply(language, tracking_summary)
        memory.add_assistant_message(reply)
        return ChatResponse(reply=reply, conversation_id=conversation_id, language=language)

    if _contains_any(
        user_input,
        {
            "exercise",
            "exercises",
            "muscle",
            "workout",
            "train",
            "تمرين",
            "تمارين",
            "اتمرن",
            "تمرن",
            "كيفية التمرين",
            "عضلة",
            "عضلات",
            "الصدر",
            "الظهر",
            "الكتف",
            "الأكتاف",
            "الأرجل",
            "الرجل",
            "الساق",
            "البطن",
        },
    ):
        reply = _exercise_reply(user_input, language)
        memory.add_assistant_message(reply)
        return ChatResponse(
            reply=reply,
            conversation_id=conversation_id,
            language=language,
            action="exercise_results",
            data={"redirect_to": "/workouts"},
        )

    llm_reply = _general_llm_reply(
        user_message=user_input,
        language=language,
        profile=profile,
        tracking_summary=tracking_summary,
        memory=memory,
        state=state,
        recent_messages=recent_messages,
    )
    if llm_reply.startswith("Ollama error:"):
        llm_reply = _lang_reply(
            language,
            "Local AI model is temporarily unavailable. Please make sure Ollama is running, then try again.",
            "نموذج الذكاء المحلي غير متاح مؤقتًا. تأكد من تشغيل Ollama ثم أعد المحاولة.",
            "نموذج الذكاء المحلي واقف مؤقتًا. شغّل Ollama وارجع جرّب.",
        )
    filtered_reply, _ = MODERATION.filter_content(llm_reply, language=language)
    memory.add_assistant_message(filtered_reply)
    return ChatResponse(reply=filtered_reply, conversation_id=conversation_id, language=language)


async def _voice_llm_responder(
    transcript: str,
    language: str,
    user_id: Optional[str],
    conversation_id: Optional[str],
) -> tuple[str, Optional[str]]:
    chat_req = ChatRequest(
        message=transcript,
        user_id=user_id,
        conversation_id=conversation_id,
        language=language,
    )
    chat_resp = await chat(chat_req)
    return chat_resp.reply, chat_resp.conversation_id


@app.post("/voice-chat", response_model=VoiceChatResponse)
async def voice_chat(
    audio: UploadFile = File(...),
    language: str = Form("en"),
    user_id: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
) -> VoiceChatResponse:
    uid = _normalize_user_id(user_id)
    conv_id = _normalize_conversation_id(conversation_id, uid)
    lang = "ar" if (language or "").lower().startswith("ar") else "en"

    if audio.content_type and not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an audio format.")

    suffix = Path(audio.filename or "").suffix.lower() or ".wav"
    input_audio_path = STATIC_AUDIO_DIR / f"input_{uuid.uuid4().hex}{suffix}"

    try:
        with input_audio_path.open("wb") as out_file:
            shutil.copyfileobj(audio.file, out_file)

        result: VoicePipelineResult = await VOICE_PIPELINE.run(
            audio_path=input_audio_path,
            language=lang,
            user_id=uid,
            conversation_id=conv_id,
            llm_responder=_voice_llm_responder,
        )

        return VoiceChatResponse(
            transcript=result.transcript,
            reply=result.reply_text,
            audio_path=result.audio_url,
            conversation_id=result.conversation_id or conv_id,
            language=lang,
        )
    except VoicePipelineError as exc:
        logger.warning("VOICE_CHAT_PIPELINE_ERROR user=%s conv=%s msg=%s", uid, conv_id, str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("VOICE_CHAT_UNKNOWN_ERROR user=%s conv=%s", uid, conv_id)
        raise HTTPException(status_code=500, detail="Voice chat failed unexpectedly.") from exc
    finally:
        try:
            audio.file.close()
        except Exception:
            pass
        try:
            input_audio_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/plans/{plan_id}/approve")
def approve_plan(plan_id: str, req: PlanActionRequest | None = None) -> dict[str, Any]:
    record = PENDING_PLANS.get(plan_id)
    if not record:
        raise HTTPException(status_code=404, detail="Plan not found")

    if req and req.user_id and record["user_id"] != req.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to approve this plan")

    record["approved"] = True
    return {
        "status": "approved",
        "plan_id": plan_id,
        "approved_plan": {
            "type": record["plan_type"],
            "plan": record["plan"],
        },
        "message": "Plan approved successfully.",
    }


@app.post("/plans/{plan_id}/reject")
def reject_plan(plan_id: str, req: PlanActionRequest | None = None) -> dict[str, Any]:
    record = PENDING_PLANS.get(plan_id)
    if not record:
        raise HTTPException(status_code=404, detail="Plan not found")

    if req and req.user_id and record["user_id"] != req.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to reject this plan")

    record["approved"] = False
    return {"status": "rejected", "plan_id": plan_id}


@app.get("/conversation/{conversation_id}")
def get_conversation_history(conversation_id: str, user_id: Optional[str] = None) -> dict[str, Any]:
    uid = _normalize_user_id(user_id)
    key = _session_key(uid, _normalize_conversation_id(conversation_id, uid))
    memory = MEMORY_SESSIONS.get(key)
    return {
        "conversation_id": conversation_id,
        "user_id": uid,
        "messages": memory.short_term.get_full_history() if memory else [],
    }


@app.post("/conversation/{conversation_id}/clear")
def clear_conversation(conversation_id: str, user_id: Optional[str] = None) -> dict[str, Any]:
    uid = _normalize_user_id(user_id)
    key = _session_key(uid, _normalize_conversation_id(conversation_id, uid))
    if key in MEMORY_SESSIONS:
        MEMORY_SESSIONS[key].clear_short_term()
    return {"status": "cleared", "conversation_id": conversation_id}


@app.get("/progress/{user_id}")
def get_progress(user_id: str) -> dict[str, Any]:
    state = _get_user_state(_normalize_user_id(user_id))
    return {
        "user_id": user_id,
        "date": datetime.utcnow().isoformat(),
        "summary": state.get("last_progress_summary", {}),
    }
