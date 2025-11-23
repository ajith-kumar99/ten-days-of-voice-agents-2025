import logging
import os
import json
import datetime
from pathlib import Path

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

from livekit.plugins import (
    murf,
    google,
    deepgram,
    silero,
    noise_cancellation,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


# --------------------------------------------------------------------
# Assistant Persona
# --------------------------------------------------------------------
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a friendly barista for Murf Coffee Roasters. The user orders by voice.

Your job is to collect one drink order with these fields (internally):
- drinkType
- size
- milk
- extras
- name

Conversation rules:
- Ask for ONE field at a time in this order: drinkType → size → milk → extras → name.
- At each step, briefly remind the options:
  • DRINK TYPE: latte, cappuccino, espresso, americano, cold brew.
  • SIZE: small, medium, large.
  • MILK: whole, skim, oat, almond, soy, or no milk.
  • EXTRAS: extra shot, vanilla syrup, caramel, whipped cream, sugar-free syrup, or no extras.
- If an answer is unclear, ask a short, polite follow-up.
- Never show JSON, tools, files, or internal state to the user.
- Keep responses short, natural, and friendly.

Tool usage:
- Whenever the user provides or changes a field, call update_order(field, value).
- When the order is complete and saved, give a short spoken confirmation
  and always end with exactly:
  "Thanks for ordering, have a nice day."
"""
        )

        # Internal order state (not spoken)
        self.order_state = {
            "drinkType": "",
            "size": "",
            "milk": "",
            "extras": [],
            "name": "",
        }

    # ----------------------------------------------------------------
    # UPDATE ORDER TOOL
    # ----------------------------------------------------------------
    @function_tool
    async def update_order(self, context: RunContext, field: str, value: str):
        """
        Update one field of the order and save to JSON when complete.

        Args:
            field: one of 'drinkType', 'size', 'milk', 'extras', 'name'
            value: user-provided value (for extras can be comma-separated)
        """

        allowed = {"drinkType", "size", "milk", "extras", "name"}
        if field not in allowed:
            return f"Unknown field '{field}'. Allowed fields: drinkType, size, milk, extras, name."

        # Extras may be comma-separated
        if field == "extras":
            parts = [e.strip() for e in value.split(",") if e.strip()]
            if parts:
                self.order_state["extras"].extend(parts)
        else:
            self.order_state[field] = value.strip()

        # Determine missing fields
        missing = []
        if not self.order_state["drinkType"]:
            missing.append("drinkType")
        if not self.order_state["size"]:
            missing.append("size")
        if not self.order_state["milk"]:
            missing.append("milk")
        if not self.order_state["extras"]:
            missing.append("extras")
        if not self.order_state["name"]:
            missing.append("name")

        # If complete → save file under src/orders/
        if not missing:
            base_dir = Path(__file__).resolve().parent  # e.g. backend/src
            orders_dir = base_dir / "orders"
            orders_dir.mkdir(exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = orders_dir / f"order_{timestamp}.json"

            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(self.order_state, f, indent=2)
                logger.info("Saved coffee order to %s", filename)
            except Exception as e:
                logger.error("Order complete but failed to save: %s", e)
                return f"Order complete but failed to save: {e}"

            # This text is for the LLM to see; you will speak your own confirmation.
            return "Order saved successfully."

        # Still missing fields → return status for the LLM to reason about
        return f"Updated '{field}'. Missing fields: {', '.join(missing)}."

    @function_tool
    async def get_order(self, context: RunContext):
        """Return current internal order state (for debugging / tools)."""
        return self.order_state


# --------------------------------------------------------------------
# PREWARM – load Silero VAD with more sensitive settings
# --------------------------------------------------------------------
def prewarm(proc: JobProcess):
    # Tuned Silero VAD to better pick up softer speech
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.35,   # more sensitive than default 0.5
        min_speech_duration=0.10,    # start speech a bit faster
        min_silence_duration=0.45,   # small pause before ending a turn
    )


# --------------------------------------------------------------------
# ENTRYPOINT
# --------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        # Deepgram STT, tuned for better transcripts
        stt=deepgram.STT(
            model="nova-3",
            language="en-US",
            detect_language=False,
            interim_results=True,
            punctuate=True,
            smart_format=True,
        ),

        llm=google.LLM(model="gemini-2.5-flash"),

        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),

        # Better VAD + turn detector combo
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),

        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info("Usage summary: %s", usage_collector.get_summary())

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
