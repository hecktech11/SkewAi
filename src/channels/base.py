"""Channel abstraction — the orchestrator talks only to `ChannelAdapter`.

Agents never know voice exists. The browser does STT (Web Speech API) and TTS
(`speechSynthesis`); the server only sees text over a WebSocket. A real
telephony adapter (Twilio) can be added later without touching the agents.

Design notes
------------
- The adapter is a thin translation layer between the orchestrator's
  `OrchestratorHooks` and a concrete transport (WebSocket, in-process, stdio).
- `capabilities()` lets the console/widget adapt (e.g. whether barge-in or
  server-side TTS is supported).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChannelCapabilities:
    """What a channel can do. The widget/console adapt their UI to these."""

    voice: bool = False                # server can expect audio-capable client
    text: bool = True                  # text turns supported
    barge_in: bool = False             # client supports barge-in detection
    server_tts: bool = False           # server synthesizes audio (False = client TTS)
    supervisor_takeover: bool = True   # supervisor can take over on this channel
    # 10/10 voice contract (defaults keep old adapters compatible):
    dtmf: bool = False                 # DTMF fallback ("press 1 if…") supported
    asr_confidence: bool = False       # channel supplies per-entity confidence
    silence_timeout_ms: int = 6000     # silence before reprompt/handoff
    tts_budget_ms: int = 1500          # per-turn TTS latency budget


class ChannelAdapter(ABC):
    """Protocol for a customer-facing channel.

    The orchestrator emits output through `send_*`; inbound customer turns
    arrive via the transport and are handed to `orchestrator.handle_customer_turn`
    by the API/gateway layer (the adapter itself does not drive the orchestrator).
    """

    channel_name: str = "base"

    @abstractmethod
    def capabilities(self) -> ChannelCapabilities:
        """Return this channel's capabilities."""
        ...

    @abstractmethod
    async def send_turn(self, text: str, *, speaker: str = "agent", meta: dict[str, Any] | None = None) -> None:
        """Deliver an agent/supervisor turn to the customer."""
        ...

    @abstractmethod
    async def send_activity(self, payload: dict[str, Any]) -> None:
        """Deliver an agent-activity card (console/monitoring)."""
        ...

    @abstractmethod
    async def send_slots_update(self, slots: dict[str, str]) -> None:
        """Deliver the current slot frame (pack-labeled)."""
        ...

    @abstractmethod
    async def send_handoff_offer(self) -> None:
        """Notify the customer that a human handoff was offered."""
        ...

    @abstractmethod
    async def send_interaction_ended(self, payload: dict[str, Any]) -> None:
        """Notify that the interaction ended (case id, audit pending)."""
        ...

    @abstractmethod
    async def hangup(self) -> None:
        """Terminate the channel (customer hung up / server closed)."""
        ...

    async def send_control(self, payload: dict[str, Any]) -> None:
        """Deliver an out-of-band control/telemetry frame (frustration, latency,
        consent prompts).

        Concrete-by-default: only channels with a structured frame surface (a
        WebSocket widget) can carry these. SMS, email and telephony have no
        place to put them, so the base ignores them rather than forcing every
        adapter to stub the method out.
        """
        return None
