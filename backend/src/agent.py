# agent.py
import logging
import random
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

logger = logging.getLogger("game-master")
logger.setLevel(logging.INFO)
load_dotenv(".env.local")


# --------------------------------------------------------------------
# Simple Game Master (GM) Agent - improved/save-after-each-output
# --------------------------------------------------------------------
class GameMaster(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are 'Nexus', a short-form Game Master (GM) for a fast-paced cyberpunk
mini-adventure. Keep responses concise (1-3 sentences) and always end
with a direct prompt for the player, e.g. "What do you do?"

Behavior:
- Start by describing the opening rooftop scene and ask the player what they do.
- If the player attempts a risky action, use roll_dice(skill, dc) tool to resolve it.
- Track a tiny world state (player name, hp, inventory, location, events).
- Persist the world state to disk after each GM output and after any state change.
- Keep the game short and easy to follow.
"""
        )

        # small in-memory game state
        self.game_state = self._load_or_default_state()

    # ---------------------
    # state helpers
    # ---------------------
    def _state_file_candidates(self):
        # Prefer local file in same folder as agent.py
        base = Path(__file__).resolve().parent
        return [
            base / "game_state.json",
            base / "shared-data" / "game_state.json",
            base.parent / "shared-data" / "game_state.json",
        ]

    def _load_or_default_state(self):
        for p in self._state_file_candidates():
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    logger.info("Loaded game state from %s", p)
                    return data
                except Exception as e:
                    logger.warning("Failed to load game state from %s: %s", p, e)

        # default state
        return {
            "meta": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_saved": None,
            },
            "player": {
                "name": "Cipher",
                "hp": 10,
                "inventory": ["short sword", "torch"],
                "location": "rooftop",
            },
            "locations": {
                "rooftop": {
                    "title": "Rooftop Over Arasaka Tower",
                    "description": "A wet rooftop under neon rain. You can see a maintenance hatch; a drone patrols above.",
                    "visited": True,
                }
            },
            "events": [],
        }

    def _persist_state_now(self) -> bool:
        """Write current in-memory game_state to the first writable candidate.
           Returns True if saved, False otherwise."""
        saved = False
        for p in self._state_file_candidates():
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                # update last_saved
                self.game_state["meta"]["last_saved"] = datetime.now(timezone.utc).isoformat()
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(self.game_state, f, indent=2, ensure_ascii=False)
                logger.info("Game state written to %s", p)
                saved = True
                break
            except Exception as e:
                logger.warning("Failed to save game state to %s: %s", p, e)
        if not saved:
            logger.error("Unable to persist game state to any candidate path.")
        return saved

    # ---------------------
    # tool: roll dice (d20)
    # ---------------------
    @function_tool
    async def roll_dice(self, context: RunContext, skill: str, dc: int = 15):
        """Roll a d20 and return a short result string. Also update events & persist."""
        roll = random.randint(1, 20)
        total = roll  # no modifiers for simplicity
        result = "failure"
        if roll == 20:
            result = "critical_success"
        elif roll == 1:
            result = "critical_failure"
        elif total >= dc:
            result = "success"
        elif total >= dc - 3:
            result = "partial"
        else:
            result = "failure"

        # update events
        event = {
            "time": datetime.now(timezone.utc).isoformat(),
            "type": "skill_check",
            "skill": skill,
            "dc": dc,
            "roll": roll,
            "result": result,
        }
        self.game_state.setdefault("events", []).append(event)

        # persist state immediately
        saved = self._persist_state_now()
        note = "saved" if saved else "not_saved"
        return f"{skill} check: roll {roll} vs DC {dc} → {result}. ({note})"

    # ---------------------
    # tool: get state (for LLM / tooling)
    # ---------------------
    @function_tool
    async def get_game_state(self, context: RunContext):
        return self.game_state

    # ---------------------
    # tool: reset state
    # ---------------------
    @function_tool
    async def reset_game_state(self, context: RunContext):
        self.game_state = self._load_or_default_state()
        self._persist_state_now()
        return "Game state reset and saved."


# --------------------------------------------------------------------
# PREWARM: load Silero VAD
# --------------------------------------------------------------------
def prewarm(proc: JobProcess):
    # silero tuned slightly more sensitive so it catches quieter speech
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.38,
        min_speech_duration=0.08,
        min_silence_duration=0.35,
    )


# --------------------------------------------------------------------
# ENTRYPOINT
# --------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # STT tuned for real-time conversation
    stt = deepgram.STT(
        model="nova-3",
        language="en-US",
        interim_results=True,
        punctuate=True,
        smart_format=True,
    )

    # Murf TTS - clear narrator voice and style
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
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info("Usage summary: %s", usage_collector.get_summary())

    ctx.add_shutdown_callback(log_usage)

    # Create assistant and start session
    assistant = GameMaster()

    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Immediately send an opening "initialize story" instruction so user hears prompt
    try:
        await session.generate_reply(
            instructions=(
                "Initialize the short rooftop scene: describe the wet rooftop, the "
                "maintenance hatch, and a patrolling drone. Ask the player: 'What do you do?'"
            )
        )
        # persist state right after the GM initial output
        assistant._persist_state_now()
    except Exception as e:
        logger.exception("Failed to generate opening reply: %s", e)

    # connect
    await ctx.connect()


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
