"""
ConvI — Conversation Normalizer
================================
Merge point for AUDIO and TEXT pipelines.

Takes either:
  - List[SpeechSegment]  (from speech pipeline / audio path)
  - raw text transcript  (from text pipeline / text path)

Outputs:
  - List[ConversationTurn]  (unified timeline, normalized to English)

Role classification uses simple heuristics + turn ordering:
  - SPEAKER_00 → agent (first speaker in a support call)
  - SPEAKER_01 → customer
  - For text transcripts: lines prefixed "Agent:" / "Customer:"
    are parsed directly; otherwise alternating turns are assumed.
"""

from __future__ import annotations

import re
from typing import List, Optional
from loguru import logger

from app.schemas import ConversationTurn, Role
from app.speech_pipeline.schemas import SpeechSegment


# ── Role assignment ───────────────────────────────────────────────────────────

_SPEAKER_ROLE_MAP: dict[str, Role] = {}   # populated dynamically per call

def _assign_roles_from_speakers(segments: List[SpeechSegment]) -> dict[str, Role]:
    """
    In a banking support call the first speaker is typically the agent.
    Map SPEAKER_00 → agent, all others → customer.
    """
    speakers_seen: list[str] = []
    for seg in segments:
        if seg.speaker_id not in speakers_seen:
            speakers_seen.append(seg.speaker_id)

    role_map: dict[str, Role] = {}
    for i, spk in enumerate(speakers_seen):
        role_map[spk] = Role.agent if i == 0 else Role.customer
    return role_map


# ── Audio path normalizer ─────────────────────────────────────────────────────

def normalize_from_speech(
    segments: List[SpeechSegment],
) -> List[ConversationTurn]:
    """
    Convert List[SpeechSegment] → List[ConversationTurn].

    Parameters
    ----------
    segments : List[SpeechSegment]
        Output from run_speech_pipeline().

    Returns
    -------
    List[ConversationTurn]
        Chronologically ordered unified conversation timeline.
    """
    if not segments:
        logger.warning("[Normalizer] No speech segments to normalize.")
        return []

    role_map = _assign_roles_from_speakers(segments)
    turns: List[ConversationTurn] = []

    for seg in segments:
        role = role_map.get(seg.speaker_id, Role.unknown)

        # For multilingual support: English stays as-is;
        # non-English text is kept in original — LLM handles translation context
        normalized_en = seg.original_text  # TODO: plug in translation if needed

        turns.append(ConversationTurn(
            speaker_id=seg.speaker_id,
            role=role,
            original_text=seg.original_text,
            normalized_text_en=normalized_en,
            language=seg.language,
            emotion=seg.emotion,
            start_time=seg.start_time,
            end_time=seg.end_time,
        ))

    logger.info(
        f"[Normalizer] Audio → {len(turns)} turns | "
        f"speakers: {set(role_map.keys())} | "
        f"roles: {set(role_map.values())}"
    )
    return turns


# ── Text path normalizer ──────────────────────────────────────────────────────

_AGENT_PREFIX    = re.compile(r"^(Agent|AGENT|Support|CSR|Representative)\s*[:\-]\s*", re.I)
_CUSTOMER_PREFIX = re.compile(r"^(Customer|CUSTOMER|Client|User|Caller)\s*[:\-]\s*", re.I)
_SPEAKER_PREFIX  = re.compile(r"^(SPEAKER_\d+)\s*[:\-]\s*", re.I)


def normalize_from_text(
    transcript: str,
    language: str = "en",
) -> List[ConversationTurn]:
    """
    Parse a raw text transcript → List[ConversationTurn].

    Supported formats (auto-detected):
      1. "Agent: ..."  / "Customer: ..."  explicit labels
      2. "SPEAKER_00: ..." / "SPEAKER_01: ..." diarized labels
      3. Plain alternating lines (assumed agent first)

    Parameters
    ----------
    transcript : str
        Raw multi-line conversation text.
    language : str
        Language code for the transcript (default: "en").

    Returns
    -------
    List[ConversationTurn]
    """
    lines = [l.strip() for l in transcript.strip().splitlines() if l.strip()]
    turns: List[ConversationTurn] = []

    for i, line in enumerate(lines):
        # Detect role from prefix
        role = Role.unknown
        text = line

        if _AGENT_PREFIX.match(line):
            role = Role.agent
            text = _AGENT_PREFIX.sub("", line).strip()
            speaker_id = "AGENT"
        elif _CUSTOMER_PREFIX.match(line):
            role = Role.customer
            text = _CUSTOMER_PREFIX.sub("", line).strip()
            speaker_id = "CUSTOMER"
        elif m := _SPEAKER_PREFIX.match(line):
            speaker_id = m.group(1).upper()
            text = _SPEAKER_PREFIX.sub("", line).strip()
            # SPEAKER_00 → agent, rest → customer
            role = Role.agent if speaker_id == "SPEAKER_00" else Role.customer
        else:
            # Plain alternating: even index → agent, odd → customer
            role = Role.agent if i % 2 == 0 else Role.customer
            speaker_id = "AGENT" if role == Role.agent else "CUSTOMER"

        if not text:
            continue

        turns.append(ConversationTurn(
            speaker_id=speaker_id,
            role=role,
            original_text=text,
            normalized_text_en=text,
            language=language,
            emotion=None,
            start_time=None,
            end_time=None,
        ))

    logger.info(
        f"[Normalizer] Text → {len(turns)} turns | lang={language}"
    )
    return turns


# ── Convenience: full text of conversation for LLM ───────────────────────────

def turns_to_dialogue_string(turns: List[ConversationTurn]) -> str:
    """
    Render ConversationTurn list as a readable dialogue string
    for feeding into the LLM prompt.
    """
    lines = []
    for t in turns:
        role_label = t.role.value.capitalize()
        emotion_tag = f" [{t.emotion}]" if t.emotion else ""
        lines.append(f"{role_label}{emotion_tag}: {t.normalized_text_en}")
    return "\n".join(lines)
