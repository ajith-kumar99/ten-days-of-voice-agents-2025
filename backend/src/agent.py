# agent.py
import logging
import os
import json
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, google, deepgram, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
load_dotenv(".env.local")


# --------------------------------------------------------------------
# Fraud Alert Voice Agent Persona (Day 6)
# --------------------------------------------------------------------
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a calm, professional Fraud Alert Representative for a fictional
bank called "SafeTrust Bank". This is a demo agent and you will never ask
for real card numbers, PINs, passwords, or any other sensitive credentials.

When a fraud alert session begins:
- Greet the user and explain you are calling about a suspicious transaction.
- Ask the user's name to locate the case.
- Verify identity using one non-sensitive security question from the case
  (for example: last merchant, last 4 digits masked, or a pre-set security phrase).
- If verification passes: read out the suspicious transaction details:
  merchant, amount, masked card, approximate time/category/source.
- Ask the user: "Did you make this transaction?" Expect a yes/no answer.
- If user confirms -> mark the case as confirmed_safe and say what happens:
  e.g. "Thanks — we'll mark this as safe."
- If user denies -> mark the case as confirmed_fraud and describe mock actions:
  e.g. "We'll block the card and open a dispute. Our fraud team will follow up."
- If verification fails -> set status verification_failed and end politely.

Persistence:
- Fraud cases are stored in a local JSON file (fraud_cases.json).
- Use the tools get_case_by_name(username) and update_case_status(case_id, status, note)
  to read and write the database.
- Do not mention file names or JSON to the user. Speak only plain, friendly sentences.

Safety:
- Use only fake/demo data.
- Do not request or process real sensitive data.

Keep replies short, professional, and reassuring.
"""
        )

        # in-memory cases and the file path we loaded from (set by _load_cases)
        self._cases = []
        self._cases_file = None
        self._cases = self._load_cases()

    # -------------------------
    # Internal: load fraud cases
    # -------------------------
    def _load_cases(self):
        base_dir = Path(__file__).resolve().parent
        # Candidate locations (prefer local file in same dir)
        candidates = [
            base_dir / "fraud_cases.json",
            base_dir.parent / "shared-data" / "fraud_cases.json",
            base_dir / "shared-data" / "fraud_cases.json",
        ]

        for p in candidates:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    logger.info("Loaded fraud cases from %s", p)
                    # store the path we loaded from so updates write back here
                    self._cases_file = p
                    return data if isinstance(data, list) else []
                except Exception as e:
                    logger.error("Failed to load fraud cases from %s: %s", p, e)

        # If none found, create local empty file and use that
        fallback = base_dir / "fraud_cases.json"
        try:
            if not fallback.exists():
                fallback.parent.mkdir(parents=True, exist_ok=True)
                with open(fallback, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=2)
                logger.info("Created new fraud cases file at %s", fallback)
            self._cases_file = fallback
            return []
        except Exception as e:
            logger.error("Failed to create fallback fraud_cases.json: %s", e)
            self._cases_file = None
            return []

    # -------------------------
    # TOOL: get a fraud case by username
    # -------------------------
    @function_tool
    async def get_case_by_name(self, context: RunContext, username: str):
        """
        Return the most recent pending fraud case for the given username.
        If none found, return the string 'no_case'.
        """
        username_norm = (username or "").strip().lower()
        if not username_norm:
            return "no_case"

        matches = [c for c in self._cases if c.get("userName", "").strip().lower() == username_norm]
        if not matches:
            return "no_case"

        # return the most recent by transactionTime if present
        def ts_key(c):
            t = c.get("transactionTime")
            if not t:
                return datetime.min
            # try common formats: already ISO or "YYYY-MM-DD HH:MM" etc.
            try:
                # Try ISO first
                return datetime.fromisoformat(t)
            except Exception:
                try:
                    # fallback parse common format
                    return datetime.strptime(t, "%Y-%m-%d %H:%M")
                except Exception:
                    return datetime.min

        matches.sort(key=ts_key, reverse=True)
        return matches[0]

    # -------------------------
    # TOOL: update case status and persist to disk
    # -------------------------
    @function_tool
    async def update_case_status(self, context: RunContext, case_id: str, status: str, note: str):
        """
        Update case status and persist to disk.
        case_id: can be the securityIdentifier or caseId (string)
        status: one of confirmed_safe, confirmed_fraud, verification_failed
        note: short note to append
        Returns:
            - "case_updated" on success
            - "no_such_case" if not found
            - "failed_to_save" if write fails
        """
        if not case_id:
            return "no_such_case"

        # try to find by matching securityIdentifier OR caseId
        target = None
        for c in self._cases:
            sec = str(c.get("securityIdentifier", "")).strip()
            cid = str(c.get("caseId", "")).strip()
            if sec and sec == str(case_id).strip():
                target = c
                break
            if cid and cid == str(case_id).strip():
                target = c
                break

        if not target:
            logger.warning("update_case_status: no case matched id=%s", case_id)
            return "no_such_case"

        # update in-memory
        target["status"] = status
        # append or set outcome note
        prev_note = target.get("notes") or target.get("outcome_note") or ""
        # keep previous notes and append short new one
        combined_note = (prev_note + " | " if prev_note else "") + (note or "")
        target["notes"] = combined_note
        target["last_updated"] = datetime.now(timezone.utc).isoformat()

        # decide path to write to
        write_path = self._cases_file
        if write_path is None:
            # fallback local path next to agent.py
            write_path = Path(__file__).resolve().parent / "fraud_cases.json"

        try:
            write_path.parent.mkdir(parents=True, exist_ok=True)
            with open(write_path, "w", encoding="utf-8") as f:
                json.dump(self._cases, f, indent=2, ensure_ascii=False)
            logger.info("Updated fraud cases written to %s", write_path)
            return "case_updated"
        except Exception as e:
            logger.error("Failed to persist updated fraud cases to %s: %s", write_path, e)
            return "failed_to_save"

    # -------------------------
    # TOOL: list all cases (for debugging)
    # -------------------------
    @function_tool
    async def list_cases(self, context: RunContext):
        # return a small view (don't leak sensitive details if any)
        safe_list = []
        for c in self._cases:
            safe_list.append({
                "caseId": c.get("caseId"),
                "userName": c.get("userName"),
                "transactionName": c.get("transactionName"),
                "transactionTime": c.get("transactionTime"),
                "status": c.get("status"),
            })
        return safe_list


# --------------------------------------------------------------------
# PREWARM: load VAD
# --------------------------------------------------------------------
def prewarm(proc: JobProcess):
    # Silero VAD tuned slightly sensitive for demo environment
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.40,
        min_speech_duration=0.08,
        min_silence_duration=0.4,
    )


# --------------------------------------------------------------------
# ENTRYPOINT
# --------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Configure STT for accurate capture
    stt = deepgram.STT(
        model="nova-3",
        language="en-US",
        interim_results=False,
        punctuate=True,
        smart_format=True,
    )

    # Murf TTS voice for agent (friendly, calm)
    tts = murf.TTS(
        voice="en-US-matthew",
        style="Conversation",
        tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=1),
        text_pacing=True,
    )

    session = AgentSession(
        stt=stt,
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=tts,
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )

    # metrics
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info("Usage summary: %s", usage_collector.get_summary())

    ctx.add_shutdown_callback(log_usage)

    # Start session with the fraud assistant
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
