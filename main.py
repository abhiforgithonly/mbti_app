import json
import os
import random
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="for meowrii ♡")

MODEL_DIR = Path("mbti_16p_models")

# ==============================================================================
# 1. LOAD TRAINED MODELS & ALIGNED QUESTIONS
# ==============================================================================
try:
    MODELS = {
        "IE": joblib.load(MODEL_DIR / "ie_model.joblib"),
        "NS": joblib.load(MODEL_DIR / "ns_model.joblib"),
        "TF": joblib.load(MODEL_DIR / "tf_model.joblib"),
        "JP": joblib.load(MODEL_DIR / "jp_model.joblib"),
    }
    with open(MODEL_DIR / "questions_bank.json", "r", encoding="utf-8") as f:
        QUESTIONS_BANK = json.load(f)
except Exception as e:
    raise RuntimeError(
        f"Failed to load models from '{MODEL_DIR}'. Make sure mbti_16p_models is in this folder. Error: {e}"
    )

MODEL_FEATURE_COLS = list(MODELS["IE"].feature_names_in_)

DISPLAY_QUESTIONS = QUESTIONS_BANK.copy()
random.Random(42).shuffle(DISPLAY_QUESTIONS)

# ==============================================================================
# 2. NATURAL, WARM & SWEET TYPE NOTES (ALL 16 TYPES)
# ==============================================================================
TYPE_PROFILES = {
    "INTJ": (
        "quiet little mastermind",
        "You act all cool and unbothered on the outside, yet you secretly care way more than you let on. You notice tiny details nobody else catches, overthink things at 2 AM, and somehow always have a plan for everything. Honestly, being let into your inner circle is the best feeling ever.",
    ),
    "INTP": (
        "sleepy daydreamer with 90 tabs open",
        "Your brain is constantly wandering off into the funniest, most random rabbit holes, and you always have the best late-night thoughts. You need your quiet recharge time, you hate fake small talk, and when you actually open up to someone, it is pure gold.",
    ),
    "ENTJ": (
        "tiny boss of the universe",
        "You have main-character energy without even trying. When you care about something—or someone—you show up 100%, take charge when things get messy, and still somehow have the softest side when it is just the two of us.",
    ),
    "ENTP": (
        "lovable little troublemaker",
        "You will start a playful debate over literally anything just to see people smile, and your brain moves at a hundred miles an hour. Life is never boring around you, and underneath all the teasing you actually have the biggest heart.",
    ),
    "INFJ": (
        "soft soul who reads minds",
        "You feel everything so deeply and always know when someone's mood shifts before they even say a word. You need plenty of cozy alone time to recharge, and the way you care about the people you love is genuinely one of a kind.",
    ),
    "INFP": (
        "sweetest daydreamer alive",
        "You get attached to stray animals, sad songs, and tiny moments that most people walk right past. You have the gentlest heart, the funniest random thoughts, and loving you feels like finding a warm blanket on a rainy day.",
    ),
    "ENFJ": (
        "walking warm hug",
        "You somehow make everyone around you feel safe, heard, and special without even trying. You remember the little things people mention once in passing, and you pour so much warmth into the people you love.",
    ),
    "ENFP": (
        "human sunshine with zero chill",
        "You get excited over the cutest little things, jump between five random ideas in one sentence, and bring so much life wherever you go. Being around you makes even an ordinary errand feel like a mini adventure.",
    ),
    "ISTJ": (
        "cozy, reliable & secretly funny",
        "You like things done properly, hate when people are late, and secretly keep track of every little detail so life runs smoothly. You don't make a big show of your feelings—you just show up every single time it matters.",
    ),
    "ISFJ": (
        "sweetest caretaker with secret sass",
        "You remember everyone's favorite snacks, notice when someone is tired before they do, and quietly make life softer for the people around you. Plus, your side-eye when someone acts silly is unmatched.",
    ),
    "ESTJ": (
        "organized queen with a soft spot",
        "You get things done while everyone else is still figuring out what is going on. You are fiercely loyal, weirdly funny when you relax, and the person everyone secretly relies on when things fall apart.",
    ),
    "ESFJ": (
        "everyone's favorite person",
        "You bring snacks, good vibes, and genuine care everywhere you walk. You notice when someone feels left out, love making people smile, and make the people closest to you feel truly cherished.",
    ),
    "ISTP": (
        "effortlessly cool cat",
        "You don't waste words, you stay calm when everyone else is panicking, and you have the driest, funniest sense of humor. You love your independence, which makes the moments you choose to spend together even sweeter.",
    ),
    "ISFP": (
        "cozy aesthetic soul",
        "You live in the moment, have immaculate taste in music and little things, and feel way more than you put into words. Being around you feels calm, easy, and genuinely comforting.",
    ),
    "ESTP": (
        "spontaneous bundle of fun",
        "You turn boring days into core memories, say whatever hilarious thing pops into your head, and never take life too seriously. You are bold, playful, and way sweeter than you let people see.",
    ),
    "ESFP": (
        "pure joy in human form",
        "You bring laughter into every room, notice every cute detail, and know how to turn a regular Tuesday into the best day of the week. Life is genuinely brighter with you in it.",
    ),
}


class QuizSubmission(BaseModel):
    answers: Dict[int, int]


# ==============================================================================
# 3. API ROUTES & RESPONSIVE RETRO KAWAII OS FRONTEND
# ==============================================================================
@app.get("/api/questions")
def get_questions():
    return [
        {"question_number": q["question_number"], "text": q["text"]}
        for q in DISPLAY_QUESTIONS
    ]


@app.post("/api/predict")
def predict_mbti(payload: QuizSubmission):
    if len(payload.answers) < len(QUESTIONS_BANK):
        raise HTTPException(status_code=400, detail="Please finish all the questions first!")

    row_dict = {col: 0.0 for col in MODEL_FEATURE_COLS}
    for entry in QUESTIONS_BANK:
        q_num = entry["question_number"]
        raw_ans = payload.answers.get(q_num, 0)
        row_dict[entry["model_col"]] = float(raw_ans * entry["sign_multiplier"])

    input_df = pd.DataFrame([row_dict])[MODEL_FEATURE_COLS]

    trait_pairs = [
        ("IE", "I", "E", "introvert", "extravert"),
        ("NS", "N", "S", "daydreamer", "realist"),
        ("TF", "T", "F", "head", "heart"),
        ("JP", "J", "P", "planner", "go with the flow"),
    ]

    final_type = ""
    breakdown = []
    for key, pos_l, neg_l, pos_name, neg_name in trait_pairs:
        p_pos = float(MODELS[key].predict_proba(input_df)[0][1])
        p_neg = 1.0 - p_pos
        winner = pos_l if p_pos >= 0.5 else neg_l
        final_type += winner
        breakdown.append({
            "winner": winner,
            "left_label": pos_name,
            "right_label": neg_name,
            "left_pct": round(p_pos * 100, 1),
            "right_pct": round(p_neg * 100, 1),
        })

    title, note = TYPE_PROFILES.get(
        final_type,
        ("one of a kind", "Honestly, no four letters could ever capture how cute you are."),
    )
    return {
        "mbti": final_type,
        "title": title,
        "description": note,
        "breakdown": breakdown,
    }


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover" />
  <title>for_meowrii.exe</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&family=Quicksand:wght@600;700&family=Caveat:wght@700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-cream: #fff4ed;
      --grid-line: rgba(122, 62, 54, 0.06);
      --ink: #7a3e36;
      --ink-soft: #9c665e;
      --peri: #b9c5e3;
      --peri-light: #dce3f5;
      --peach: #f8c99e;
      --blush: #f7c6c7;
      --blush-deep: #f2a4a8;
      --white: #ffffff;
    }

    * {
      box-sizing: border-box;
      -webkit-tap-highlight-color: transparent;
    }

    html, body {
      margin: 0;
      min-height: 100dvh;
      touch-action: manipulation;
    }

    body {
      background-color: var(--bg-cream);
      background-image:
        linear-gradient(to right, var(--grid-line) 1.5px, transparent 1.5px),
        linear-gradient(to bottom, var(--grid-line) 1.5px, transparent 1.5px);
      background-size: 38px 38px;
      color: var(--ink);
      font-family: 'Quicksand', sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: max(18px, env(safe-area-inset-top)) max(14px, env(safe-area-inset-right)) max(22px, env(safe-area-inset-bottom)) max(14px, env(safe-area-inset-left));
    }

    /* Desktop & iPad Grid Layout */
    .desktop {
      width: 100%;
      max-width: 860px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 185px;
      gap: 20px;
      align-items: start;
    }

    /* Retro Browser Window */
    .browser-wrap {
      position: relative;
      width: 100%;
    }

    .browser-tab {
      display: inline-flex;
      align-items: center;
      gap: 14px;
      background: var(--peri);
      border: 2.5px solid var(--ink);
      border-bottom: none;
      border-radius: 14px 14px 0 0;
      padding: 6px 16px;
      font-family: 'Courier Prime', monospace;
      font-weight: 700;
      font-size: 14px;
      color: var(--ink);
      margin-left: 16px;
      position: relative;
      z-index: 2;
    }

    .browser-window {
      background: var(--white);
      border: 2.5px solid var(--ink);
      border-radius: 18px;
      box-shadow: 6px 6px 0px rgba(122, 62, 54, 0.14);
      overflow: hidden;
      margin-top: -2px;
    }

    /* Browser Address Bar */
    .browser-toolbar {
      background: var(--peri);
      border-bottom: 2.5px solid var(--ink);
      padding: 9px 14px;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .nav-icons {
      display: flex;
      align-items: center;
      gap: 8px;
      font-family: 'Courier Prime', monospace;
      font-weight: 700;
      font-size: 16px;
      user-select: none;
    }

    .url-bar {
      flex: 1;
      min-width: 0;
      background: var(--white);
      border: 2px solid var(--ink);
      border-radius: 99px;
      padding: 4px 14px;
      font-family: 'Courier Prime', monospace;
      font-size: 13px;
      color: var(--ink);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .star-icon {
      font-size: 16px;
      color: var(--peach);
      -webkit-text-stroke: 1.5px var(--ink);
    }

    /* Inner Pastel Sky Viewport */
    .browser-body {
      padding: 18px;
      background: var(--white);
    }

    .sky-canvas {
      background: linear-gradient(180deg, #f9d1d1 0%, #fce3dc 100%);
      border: 2.5px solid var(--ink);
      border-radius: 10px;
      padding: 24px 20px;
      position: relative;
      overflow: hidden;
      min-height: 360px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }

    .sky-decor {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 0;
    }

    .content-layer {
      position: relative;
      z-index: 2;
    }

    /* Classic Dialog Box ("do you want to continue?") */
    .dialog-card {
      background: var(--white);
      border: 2.5px solid var(--ink);
      border-radius: 16px;
      padding: 24px 20px;
      text-align: center;
      max-width: 390px;
      margin: 0 auto;
      box-shadow: 4px 4px 0px rgba(122, 62, 54, 0.14);
    }

    .dialog-title {
      font-family: 'Courier Prime', monospace;
      font-weight: 700;
      font-size: 19px;
      line-height: 1.45;
      margin: 0 0 10px;
      color: var(--ink);
    }

    .dialog-sub {
      font-size: 14px;
      line-height: 1.55;
      color: var(--ink-soft);
      margin: 0 0 20px;
    }

    .dialog-btns {
      display: flex;
      justify-content: center;
      gap: 14px;
    }

    .pixel-btn {
      font-family: 'Courier Prime', monospace;
      font-weight: 700;
      font-size: 15px;
      letter-spacing: 1px;
      padding: 10px 26px;
      border: 2.5px solid var(--ink);
      border-radius: 8px;
      cursor: pointer;
      color: var(--ink);
      box-shadow: 3px 3px 0px var(--ink);
      transition: transform 0.1s ease, box-shadow 0.1s ease;
      user-select: none;
    }

    .pixel-btn:active {
      transform: translate(2px, 2px);
      box-shadow: 1px 1px 0px var(--ink);
    }

    .btn-peach { background: var(--peach); }
    .btn-peri  { background: var(--peri); }

    /* Search-Style Progress Bar Below Browser */
    .search-progress-bar {
      margin-top: 14px;
      background: var(--white);
      border: 2.5px solid var(--ink);
      border-radius: 14px;
      height: 42px;
      display: flex;
      align-items: center;
      overflow: hidden;
      box-shadow: 4px 4px 0px rgba(122, 62, 54, 0.12);
    }

    .search-track {
      flex: 1;
      height: 100%;
      padding: 6px 10px;
      display: flex;
      align-items: center;
      position: relative;
    }

    .search-fill {
      height: 100%;
      width: 0%;
      background: var(--blush);
      border-radius: 7px;
      transition: width 0.3s ease;
    }

    .search-label {
      position: absolute;
      left: 18px;
      font-family: 'Courier Prime', monospace;
      font-weight: 700;
      font-size: 13px;
      color: var(--ink);
    }

    .search-btn-icon {
      width: 48px;
      height: 100%;
      background: var(--peri);
      border-left: 2.5px solid var(--ink);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 17px;
    }

    /* Quiz Question Card inside Sky */
    .q-window {
      background: var(--white);
      border: 2.5px solid var(--ink);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 4px 4px 0px rgba(122, 62, 54, 0.14);
    }

    .q-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: 'Courier Prime', monospace;
      font-size: 12.5px;
      font-weight: 700;
      margin-bottom: 10px;
      padding-bottom: 8px;
      border-bottom: 2px dashed var(--blush-deep);
    }

    .q-text {
      font-size: 17.5px;
      font-weight: 700;
      line-height: 1.45;
      min-height: 76px;
      margin-bottom: 14px;
      color: var(--ink);
    }

    .options-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }

    button.opt-pill {
      background: var(--bg-cream);
      border: 2px solid var(--ink);
      border-radius: 10px;
      padding: 10px 13px;
      font-family: 'Quicksand', sans-serif;
      font-weight: 700;
      font-size: 14.5px;
      color: var(--ink);
      text-align: left;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 10px;
      transition: transform 0.1s ease, background-color 0.1s ease;
      user-select: none;
    }

    @media (hover: hover) {
      button.opt-pill:hover {
        background: var(--peri-light);
        transform: translateX(3px);
      }
    }

    button.opt-pill:active {
      transform: scale(0.99);
    }

    button.opt-pill.selected {
      background: var(--peach);
      box-shadow: 2px 2px 0px var(--ink);
    }

    .q-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 14px;
      gap: 10px;
    }

    /* Right Side Desktop Dock */
    .side-dock {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 16px;
    }

    .heart-bubble {
      background: var(--white);
      border: 2.5px solid var(--ink);
      border-radius: 14px;
      padding: 10px 14px;
      display: flex;
      gap: 6px;
      position: relative;
      box-shadow: 3px 3px 0px rgba(122, 62, 54, 0.12);
    }

    .heart-bubble::after {
      content: "";
      position: absolute;
      bottom: -9px;
      left: 50%;
      transform: translateX(-50%) rotate(45deg);
      width: 14px;
      height: 14px;
      background: var(--white);
      border-right: 2.5px solid var(--ink);
      border-bottom: 2.5px solid var(--ink);
    }

    .h-icon {
      font-size: 16px;
      color: var(--blush);
      -webkit-text-stroke: 1.5px var(--ink);
      transition: transform 0.2s ease, color 0.2s ease;
    }

    .h-icon.active {
      color: #ff7b89;
      transform: scale(1.15);
    }

    .dock-mid {
      display: flex;
      align-items: center;
      gap: 12px;
      width: 100%;
      justify-content: center;
    }

    .globe-icon {
      width: 58px;
      height: 58px;
      border-radius: 50%;
      background: var(--peri);
      border: 2.5px solid var(--ink);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 3px 3px 0px rgba(122, 62, 54, 0.12);
    }

    .folders-col {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .folder-tab {
      width: 68px;
      height: 44px;
      background: var(--peach);
      border: 2.5px solid var(--ink);
      border-radius: 0 8px 8px 8px;
      position: relative;
      margin-top: 7px;
      box-shadow: 3px 3px 0px rgba(122, 62, 54, 0.12);
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: 'Courier Prime', monospace;
      font-size: 10px;
      font-weight: 700;
    }

    .folder-tab::before {
      content: "";
      position: absolute;
      top: -9px;
      left: -2.5px;
      width: 28px;
      height: 8px;
      background: var(--peach);
      border: 2.5px solid var(--ink);
      border-bottom: none;
      border-radius: 6px 6px 0 0;
    }

    .selection-box {
      border: 2px dashed var(--ink);
      padding: 12px;
      width: 156px;
      position: relative;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 10px;
    }

    .corner-sq {
      width: 7px;
      height: 7px;
      background: var(--white);
      border: 2px solid var(--ink);
      position: absolute;
    }
    .tl { top: -5px; left: -5px; }
    .tr { top: -5px; right: -5px; }
    .bl { bottom: -5px; left: -5px; }
    .br { bottom: -5px; right: -5px; }

    .doc-files {
      display: flex;
      gap: 8px;
    }

    .doc-sheet {
      width: 42px;
      height: 52px;
      background: var(--white);
      border: 2px solid var(--ink);
      border-radius: 4px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: 'Courier Prime', monospace;
      font-weight: 700;
      font-size: 13px;
      position: relative;
    }

    .doc-sheet::before {
      content: "";
      position: absolute;
      top: -2px;
      right: -2px;
      width: 10px;
      height: 10px;
      background: var(--peri);
      border-left: 2px solid var(--ink);
      border-bottom: 2px solid var(--ink);
    }

    .star-badge {
      background: var(--white);
      border: 2px solid var(--ink);
      border-radius: 10px;
      padding: 6px 14px;
      font-size: 20px;
      color: var(--peach);
      -webkit-text-stroke: 1.5px var(--ink);
    }

    /* Result Screen Styling */
    .result-box {
      background: var(--white);
      border: 2.5px solid var(--ink);
      border-radius: 14px;
      padding: 20px 18px;
      box-shadow: 4px 4px 0px rgba(122, 62, 54, 0.14);
    }

    .mbti-code {
      font-family: 'Courier Prime', monospace;
      font-size: 40px;
      font-weight: 700;
      letter-spacing: 5px;
      text-align: center;
      color: var(--ink);
      background: var(--peach);
      border: 2.5px solid var(--ink);
      border-radius: 12px;
      padding: 6px 14px;
      display: inline-block;
      margin: 6px 0;
    }

    .trait-bar-row {
      margin-bottom: 10px;
    }

    .trait-bar-labels {
      display: flex;
      justify-content: space-between;
      font-family: 'Courier Prime', monospace;
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 4px;
    }

    .trait-track {
      height: 12px;
      background: var(--bg-cream);
      border: 2px solid var(--ink);
      border-radius: 99px;
      overflow: hidden;
    }

    .trait-fill {
      height: 100%;
      background: var(--peri);
      border-right: 2px solid var(--ink);
    }

    /* Interactive Envelope Love Note */
    .mail-trigger {
      margin-top: 14px;
      background: var(--peach);
      border: 2.5px solid var(--ink);
      border-radius: 12px;
      padding: 12px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      cursor: pointer;
      box-shadow: 3px 3px 0px var(--ink);
      user-select: none;
    }

    .mail-trigger:active {
      transform: translate(2px, 2px);
      box-shadow: 1px 1px 0px var(--ink);
    }

    .notif-dot {
      background: var(--white);
      border: 2px solid var(--ink);
      border-radius: 50%;
      width: 24px;
      height: 24px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-family: 'Courier Prime', monospace;
      font-weight: 700;
      font-size: 12px;
    }

    .love-letter {
      margin-top: 12px;
      background: var(--bg-cream);
      border: 2px dashed var(--ink);
      border-radius: 12px;
      padding: 14px;
      font-family: 'Caveat', cursive;
      font-size: 23px;
      line-height: 1.32;
      color: var(--ink);
    }

    .hidden { display: none !important; }

    /* Mobile & Small Tablet Responsive Adjustments */
    @media (max-width: 760px) {
      .desktop {
        grid-template-columns: 1fr;
        max-width: 500px;
        gap: 14px;
      }
      .side-dock {
        order: -1;
        flex-direction: row;
        justify-content: center;
      }
      .dock-mid, .selection-box {
        display: none;
      }
      .browser-body {
        padding: 12px;
      }
      .sky-canvas {
        padding: 18px 13px;
        min-height: 310px;
      }
      .q-window {
        padding: 15px 13px;
      }
      .q-text {
        font-size: 16.5px;
        min-height: 66px;
        margin-bottom: 12px;
      }
      button.opt-pill {
        padding: 10px 12px;
        font-size: 14px;
      }
      .nav-icons {
        gap: 6px;
        font-size: 14px;
      }
      .url-bar {
        font-size: 12px;
        padding: 4px 10px;
      }
      .mbti-code {
        font-size: 34px;
      }
    }
  </style>
</head>
<body>
  <div class="desktop">
    <!-- LEFT: RETRO BROWSER + SEARCH PROGRESS BAR -->
    <div class="browser-wrap">
      <div class="browser-tab">
        <span>for_meowrii</span>
        <span>×</span>
      </div>

      <div class="browser-window">
        <div class="browser-toolbar">
          <div class="nav-icons">
            <span>←</span>
            <span style="opacity:0.45;">→</span>
            <span>↻</span>
            <span>⌂</span>
          </div>
          <div class="url-bar" id="url-text">meowrii://little-personality-test</div>
          <div class="star-icon">★</div>
        </div>

        <div class="browser-body">
          <div class="sky-canvas">
            <!-- Pastel Clouds & Sun Vector Layer -->
            <svg class="sky-decor" viewBox="0 0 600 380" preserveAspectRatio="none">
              <circle cx="135" cy="78" r="24" fill="#F8C99E" stroke="#7A3E36" stroke-width="2.5"/>
              <path d="M-10,145 Q25,115 55,140 Q80,130 95,155 L-10,155 Z" fill="#FFFFFF" stroke="#7A3E36" stroke-width="2.5"/>
              <path d="M320,105 Q335,80 365,92 Q385,72 415,92 Q440,90 448,112 L315,112 Z" fill="#FFFFFF" stroke="#7A3E36" stroke-width="2.5"/>
              <path d="M130,340 Q150,305 185,320 Q215,295 250,320 Q280,315 295,342 L125,342 Z" fill="#FFFFFF" stroke="#7A3E36" stroke-width="2.5"/>
              <path d="M445,285 Q470,250 505,265 Q535,240 570,268 Q595,265 610,290 L440,290 Z" fill="#FFFFFF" stroke="#7A3E36" stroke-width="2.5"/>
              <text x="65" y="75" fill="#FFFFFF" font-size="18">✦</text>
              <text x="275" y="160" fill="#FFFFFF" font-size="16">✦</text>
              <text x="515" y="95" fill="#FFFFFF" font-size="20">✦</text>
            </svg>

            <div class="content-layer">
              <!-- SCREEN 1: WELCOME DIALOG -->
              <div id="intro-box" class="dialog-card">
                <p class="dialog-title">hey meowrii,<br>do you want to continue?</p>
                <p class="dialog-sub">
                  52 silly little questions to scientifically diagnose what kind of cute menace you are.
                </p>
                <div class="dialog-btns">
                  <button class="pixel-btn btn-peach" onclick="startQuiz()">YES</button>
                  <button class="pixel-btn btn-peri" onclick="teaseNo()">NO</button>
                </div>
                <p id="no-msg" style="font-size:12.5px; font-weight:700; margin:12px 0 0; color:#7a3e36;"></p>
              </div>

              <!-- SCREEN 2: QUIZ CARD -->
              <div id="quiz-box" class="q-window hidden">
                <div class="q-header">
                  <span id="counter">question 01 / 52</span>
                  <span>meowrii_scan.dat</span>
                </div>
                <div class="q-text" id="q-text"></div>
                <div class="options-grid" id="options"></div>
                <div class="q-footer">
                  <button class="pixel-btn btn-peri" style="padding:7px 14px; font-size:13px;" id="prev-btn" onclick="prevQ()">← BACK</button>
                  <button class="pixel-btn btn-peach hidden" style="padding:8px 18px; font-size:13.5px;" id="submit-btn" onclick="submitQuiz()">
                    OPEN RESULT ✉
                  </button>
                </div>
              </div>

              <!-- SCREEN 3: RESULT CARD -->
              <div id="result-box" class="result-box hidden">
                <div style="text-align:center;">
                  <div style="font-family:'Courier Prime',monospace; font-size:12px; font-weight:700;">MEOWRII'S OFFICIAL FILE</div>
                  <div class="mbti-code" id="res-mbti"></div>
                  <div style="font-weight:700; font-size:17px; margin-bottom:10px;" id="res-title"></div>
                </div>

                <p style="font-size:13.5px; line-height:1.55; background:#fff4ed; border:2px solid #7a3e36; border-radius:10px; padding:11px 13px; margin:0 0 14px;" id="res-desc"></p>

                <div id="res-bars"></div>

                <!-- Interactive Envelope Note -->
                <div class="mail-trigger" onclick="toggleLetter()">
                  <span style="font-family:'Courier Prime',monospace; font-weight:700; font-size:13.5px;">✉ tap to read note for meowrii</span>
                  <span class="notif-dot">1</span>
                </div>
                <div id="love-letter" class="love-letter hidden">
                  i made this whole silly thing just to make you smile today, meowrii. no matter what four letters pop up here, you're still my favorite person in the whole world — weird random thoughts, dramatic sighs, late-night snacks and all. love you loads 🐾♡
                </div>

                <button class="pixel-btn btn-peri" style="width:100%; margin-top:14px; font-size:13px;" onclick="location.reload()">
                  RESTART.EXE ↻
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Search-Style Progress Bar Underneath -->
      <div class="search-progress-bar">
        <div class="search-track">
          <div class="search-fill" id="progress"></div>
          <span class="search-label" id="progress-text">ready when you are, meowrii...</span>
        </div>
        <div class="search-btn-icon">🔍</div>
      </div>
    </div>

    <!-- RIGHT (DESKTOP/IPAD) / TOP (MOBILE): RETRO DESKTOP WIDGETS -->
    <div class="side-dock">
      <!-- 5 Hearts Progress Bubble -->
      <div class="heart-bubble" id="hearts-row">
        <span class="h-icon">♥</span>
        <span class="h-icon">♥</span>
        <span class="h-icon">♥</span>
        <span class="h-icon">♥</span>
        <span class="h-icon">♥</span>
      </div>

      <!-- Globe + 3 Pastel Folders -->
      <div class="dock-mid">
        <div class="globe-icon">
          <svg width="36" height="36" viewBox="0 0 36 36">
            <circle cx="18" cy="18" r="15" fill="none" stroke="#FFFFFF" stroke-width="2"/>
            <ellipse cx="18" cy="18" rx="7" ry="15" fill="none" stroke="#FFFFFF" stroke-width="2"/>
            <line x1="3" y1="18" x2="33" y2="18" stroke="#FFFFFF" stroke-width="2"/>
          </svg>
        </div>
        <div class="folders-col">
          <div class="folder-tab">cute</div>
          <div class="folder-tab">silly</div>
          <div class="folder-tab">meowrii</div>
        </div>
      </div>

      <!-- Dashed Selection Box with Aa Files & Star -->
      <div class="selection-box">
        <span class="corner-sq tl"></span>
        <span class="corner-sq tr"></span>
        <span class="corner-sq bl"></span>
        <span class="corner-sq br"></span>
        <div class="doc-files">
          <div class="doc-sheet">Aa</div>
          <div class="doc-sheet">Aa</div>
        </div>
        <div class="star-badge">★</div>
      </div>
    </div>
  </div>

  <script>
    let questions = [];
    let current = 0;
    let answers = {};

    const CHOICES = [
      { emoji: "💀", label: "hard no, never", val: -3 },
      { emoji: "🙅‍♀️", label: "nope, not me", val: -2 },
      { emoji: "🤨", label: "kinda no", val: -1 },
      { emoji: "😶", label: "depends / meh", val: 0 },
      { emoji: "👀", label: "lowkey yes", val: 1 },
      { emoji: "🌷", label: "definitely me", val: 2 },
      { emoji: "🔥", label: "100% literally me", val: 3 },
    ];

    async function loadQuestions() {
      const res = await fetch('/api/questions');
      questions = await res.json();
    }

    function teaseNo() {
      document.getElementById('no-msg').innerText = "nice try meowrii, the NO button is decorative. click YES ♡";
    }

    function startQuiz() {
      if (!questions.length) return;
      document.getElementById('intro-box').classList.add('hidden');
      document.getElementById('quiz-box').classList.remove('hidden');
      document.getElementById('url-text').innerText = "meowrii://vibe-check-in-progress";
      render();
    }

    function updateHearts(pct) {
      const hearts = document.querySelectorAll('#hearts-row .h-icon');
      const lit = Math.ceil((pct / 100) * hearts.length);
      hearts.forEach((h, idx) => h.classList.toggle('active', idx < lit));
    }

    function render() {
      const q = questions[current];
      const answeredCount = Object.keys(answers).length;
      const pct = Math.round(((current + 1) / questions.length) * 100);

      document.getElementById('counter').innerText = `question ${String(current + 1).padStart(2, '0')} / ${questions.length}`;
      document.getElementById('progress').style.width = `${pct}%`;
      document.getElementById('progress-text').innerText = `scanning meowrii... ${pct}%`;
      updateHearts(pct);

      document.getElementById('q-text').innerText = q.text;
      document.getElementById('prev-btn').disabled = current === 0;

      const optDiv = document.getElementById('options');
      optDiv.innerHTML = '';
      CHOICES.forEach(c => {
        const btn = document.createElement('button');
        btn.className = 'opt-pill' + (answers[q.question_number] === c.val ? ' selected' : '');
        btn.innerHTML = `<span>${c.emoji}</span><span>${c.label}</span>`;
        btn.onclick = () => selectAnswer(q.question_number, c.val);
        optDiv.appendChild(btn);
      });

      const isDone = answeredCount === questions.length;
      document.getElementById('submit-btn').classList.toggle('hidden', !isDone);
    }

    function selectAnswer(qNum, val) {
      answers[qNum] = val;
      if (current < questions.length - 1) {
        current++;
      }
      render();
    }

    function prevQ() {
      if (current > 0) {
        current--;
        render();
      }
    }

    function toggleLetter() {
      document.getElementById('love-letter').classList.toggle('hidden');
    }

    async function submitQuiz() {
      const btn = document.getElementById('submit-btn');
      btn.innerText = "LOADING...";
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers })
      });
      const data = await res.json();

      document.getElementById('quiz-box').classList.add('hidden');
      document.getElementById('result-box').classList.remove('hidden');
      document.getElementById('url-text').innerText = `meowrii://result/${data.mbti.toLowerCase()}`;
      document.getElementById('progress').style.width = '100%';
      document.getElementById('progress-text').innerText = 'diagnosis complete ♡';
      updateHearts(100);

      document.getElementById('res-mbti').innerText = data.mbti;
      document.getElementById('res-title').innerText = data.title;
      document.getElementById('res-desc').innerText = data.description;

      const bars = document.getElementById('res-bars');
      bars.innerHTML = '';
      data.breakdown.forEach(t => {
        bars.innerHTML += `
          <div class="trait-bar-row">
            <div class="trait-bar-labels">
              <span>${t.left_label} (${t.left_pct}%)</span>
              <span>${t.right_label} (${t.right_pct}%)</span>
            </div>
            <div class="trait-track"><div class="trait-fill" style="width:${t.left_pct}%"></div></div>
          </div>`;
      });
    }

    loadQuestions();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)