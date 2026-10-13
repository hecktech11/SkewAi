"""P2 and P3 regressions. Each test calls a shipped function."""

from __future__ import annotations

import asyncio
import struct
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.frontline.intake import IntakeSlots
from src.triage.triage_agent import TriageResult


def test_concurrent_narrate_cannot_pass_one_turn_cap(monkeypatch, reset_ops_db):
    from src.ai import provider
    from src.config import settings

    monkeypatch.setenv("FRONTLINE_LLM_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    # Settings is frozen. The cap is read from this instance at call time.
    original_cap = settings.llm_turn_cap
    object.__setattr__(settings, "llm_turn_cap", 1)
    monkeypatch.setattr(provider, "_http_chat", lambda system, user, model: "noted")

    barrier = threading.Barrier(8)
    results: list = []
    errors: list = []

    def call() -> None:
        try:
            barrier.wait(timeout=5)
            results.append(
                provider.narrate(
                    site="intake_phrasing",
                    system="sys",
                    user="user",
                    fallback="fb",
                )
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(8)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()
    finally:
        object.__setattr__(settings, "llm_turn_cap", original_cap)

    assert errors == []
    assert sum(1 for row in results if row.used_llm) == 1
    assert provider.get_spend()["turn_count"] <= 1


def test_supervisor_whisper_escapes_ssml():
    from src.voice.whisper_generator import generate_supervisor_whisper

    slots = IntakeSlots(
        entity_1="2019",
        entity_2="HONDA & CO",
        entity_3="<CR-V>",
        category="BRAKES & <STEERING>",
        description="grind",
    )
    triage = TriageResult(
        severity="Critical",
        priority="P1",
        sentiment_peak=0.2,
        safety_flag_tripped=True,
    )
    packet = generate_supervisor_whisper(
        "int_abcdef", slots, triage, ["fire & smoke"]
    )
    ssml = packet["whisper_audio_ssml"]
    ET.fromstring(ssml)
    assert "<CR-V>" not in ssml
    assert "&lt;CR-V&gt;" in ssml
    assert "fire & smoke" not in ssml
    assert "fire &amp; smoke" in ssml
    assert "BRAKES &amp; &lt;STEERING&gt;" in ssml


@pytest.mark.asyncio
async def test_activity_summary_omits_supervisor_whisper(orchestrator_factory):
    orch, hooks = orchestrator_factory()
    await orch.start()
    await orch.handle_customer_turn("the car is on fire")
    whispers = [
        row for row in hooks.activities if row.get("action_type") == "supervisor_whisper"
    ]
    assert whispers
    for row in whispers:
        summary = str(row.get("summary") or "")
        assert summary == "Supervisor coach note queued."
        assert "<speak" not in summary
        assert "Trigger words" not in summary
    assert all("<speak" not in text for text in hooks.agent_texts())


def test_iso_vin_follows_check_digit():
    from src.voice.vin_validator import validate_iso3779_vin

    assert validate_iso3779_vin("1HGCR2F85HA000000") is True
    assert validate_iso3779_vin("1HGCR2F84HA000000") is False
    assert validate_iso3779_vin("1HGCM82633A004352") is True
    assert validate_iso3779_vin("1HGCM82633A004353") is False


def test_phonetic_keeps_unknown_as_in_letter_and_xray():
    from src.voice.phonetic_normalizer import parse_spoken_alphanumerics

    assert parse_spoken_alphanumerics("1 F as in Frank T Y 2 8 Kilo") == "1FTY28K"
    assert parse_spoken_alphanumerics("Q as in Quincy") == "Q"
    assert parse_spoken_alphanumerics("B as in Baltimore") == "B"
    assert parse_spoken_alphanumerics("x-ray") == "X"
    assert (
        parse_spoken_alphanumerics(
            "1 H as in Henry Golf Charlie Romeo 2 Foxtrot 8 4 Hotel Alpha"
        )
        == "1HGCR2F84HA"
    )


@pytest.mark.asyncio
async def test_diagnostics_yields_the_event_loop(monkeypatch, reset_ops_db):
    from src.voice import benchmark_latency as bench_mod
    from src.voice.wiring import run_single_trial_diagnostics

    real = bench_mod.ulaw8k_to_pcm16k

    def slow(payload: bytes) -> bytes:
        time.sleep(0.12)
        return real(payload) if payload else b"\x00\x00"

    monkeypatch.setattr(bench_mod, "ulaw8k_to_pcm16k", slow)
    order: list[str] = []

    async def beat() -> None:
        await asyncio.sleep(0.02)
        order.append("beat")

    async def diag() -> None:
        await run_single_trial_diagnostics()
        order.append("end")

    await asyncio.gather(diag(), beat())
    assert order.index("beat") < order.index("end")


def test_vector_identity_and_exact_cosine():
    from src.ml_runtime.embedding_space import (
        EmbeddingVersionMismatch,
        as_embedded,
        compare_embeddings,
    )
    from src.ml_runtime.embeddings import cosine
    from src.ml_runtime.onnx_embedder import mean_pool

    keyed = as_embedded([1.0, 0.0], "v", model_key="hash")
    blank = as_embedded([1.0, 0.0], "v", model_key="")
    with pytest.raises(EmbeddingVersionMismatch):
        compare_embeddings(keyed, blank)

    padded = as_embedded([1.0, 0.0], "v", padding_strategy="zeropad512")
    unpadded = as_embedded([1.0, 0.0], "v", padding_strategy="")
    with pytest.raises(EmbeddingVersionMismatch):
        compare_embeddings(padded, unpadded)

    left = as_embedded([3.0, 4.0], "v")
    right = as_embedded([3.0, 4.0], "v")
    assert compare_embeddings(left, right) == 1.0
    assert cosine([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]) == 1.0
    with pytest.raises(ValueError):
        cosine([float("nan"), 1.0], [1.0, 0.0])

    hidden = np.ones((1, 2, 4), dtype=np.float32)
    mask = np.zeros((1, 2), dtype=np.int64)
    pooled = mean_pool(hidden, mask)
    assert pooled.shape == (1, 4)
    assert np.all(pooled == 0.0)


def test_novel_false_is_not_a_positive_and_unlabeled_is_excluded():
    from src.ml_runtime.embedding_eval import evaluate_pairs, per_class_scorecard
    from src.ml_runtime.hash_embedder import HashEmbedder

    embedder = HashEmbedder()
    labeled = {
        "query": "brakes failed",
        "candidate": "sunny day",
        "label": "negative",
        "eval_class": "novel_false",
    }
    card = per_class_scorecard(embedder, [labeled])
    assert card["novel_false"]["positives"] == 0

    report = evaluate_pairs(
        {"hash": embedder},
        [labeled, {"query": "a", "candidate": "b"}],
    )
    stats = report["providers"]["hash"]
    assert stats["positive_n"] == 0
    assert stats["negative_n"] == 1
    assert report["acceptance_eligible"] is False


def test_resample_state_tracks_the_whole_buffer():
    from src.voice.telephony_bridge import (
        pcm16k_to_ulaw8k,
        ulaw8k_to_pcm16k,
        ulaw8k_to_pcm16k_carrying,
    )

    pcm = struct.pack(
        "<400h",
        *[(i * 100) % 20000 - 10000 for i in range(400)],
    )
    ulaw = pcm16k_to_ulaw8k(pcm)
    assert ulaw
    mid = max(1, len(ulaw) // 2)
    whole = ulaw8k_to_pcm16k(ulaw)
    first, state = ulaw8k_to_pcm16k_carrying(ulaw[:mid], None)
    second, _state = ulaw8k_to_pcm16k_carrying(ulaw[mid:], state)
    carried = first + second
    reset = ulaw8k_to_pcm16k(ulaw[:mid]) + ulaw8k_to_pcm16k(ulaw[mid:])

    def distance(left: bytes, right: bytes) -> int:
        count = min(len(left), len(right)) // 2
        if count == 0:
            return abs(len(left) - len(right))
        a = struct.unpack(f"<{count}h", left[: count * 2])
        b = struct.unpack(f"<{count}h", right[: count * 2])
        return sum(abs(x - y) for x, y in zip(a, b)) + abs(len(left) - len(right))

    assert distance(carried, whole) <= distance(reset, whole)


def test_playback_counts_finished_words_and_one_duration_model(reset_ops_db):
    from src.ledger.journal import AgentActionJournal
    from src.voice.interruption_reconciler import SpokenPlaybackReconciler
    from src.voice.wiring import estimate_word_alignment, reconcile_barge_in

    rec = SpokenPlaybackReconciler(AgentActionJournal(), "int_audible")
    rec.register_planned_turn(
        "act_aud",
        "Please describe",
        [
            {"word": "Please", "start": 0.0, "end": 0.3},
            {"word": "describe", "start": 0.3, "end": 0.7},
        ],
    )
    rec.playback_start_ts = time.monotonic() - 0.35
    spoken = rec.handle_barge_in()
    assert "Please" in spoken
    assert "describe" not in spoken

    text = "a b c d"
    elapsed = 500
    out = reconcile_barge_in("int_align", None, text, [], elapsed)
    expected: list[str] = []
    for marker in estimate_word_alignment(text):
        end_ms = int(float(marker["end"]) * 1000)
        if end_ms <= elapsed:
            expected.append(str(marker["word"]))
        else:
            break
    assert out == f"{' '.join(expected)} [INTERRUPTED]"
    assert out.startswith("a b")


def test_redact_pii_removes_phone_and_vin():
    from src.security.pii import find_pii, redact_pii

    raw = "reach me at 415-555-0133 vin 1HGCM82633A004352"
    red = redact_pii(raw)
    assert "415-555-0133" not in red
    assert "1HGCM82633A004352" not in red
    assert "[PHONE]" in red
    assert "[VIN]" in red
    kinds = find_pii(raw)
    assert "phone" in kinds
    assert "vin" in kinds
    card = "4111111111111111"
    assert "[CARD]" in redact_pii(f"card {card}")
    assert "card" in find_pii(f"card {card}")


def test_missing_minilm_artifact_falls_back_to_rules(monkeypatch, tmp_path, reset_ops_db):
    from src.domains.loader import load_pack
    from src.ml_runtime.onnx_embedder import reset_process_onnx_embedder
    from src.ml_runtime.slm_runtime import (
        extract_category_with_slm,
        get_slm_embedder,
        predict_category_zero_shot,
        reset_slm_cache,
    )

    monkeypatch.setenv("FRONTLINE_SLM_ARTIFACT_DIR", str(tmp_path / "missing-minilm"))
    monkeypatch.delenv("FRONTLINE_EMBEDDING_TOY", raising=False)
    monkeypatch.setenv("FRONTLINE_SLM_MODE", "live")
    reset_process_onnx_embedder()
    reset_slm_cache()
    try:
        assert get_slm_embedder() is None
        pack = load_pack("automotive_nhtsa", reload=True)
        result = predict_category_zero_shot("the brakes grind when I stop", pack)
        assert result.category is None
        ctx = type("Ctx", (), {"pack": pack, "interaction_id": "int_slm_fb"})()
        _category, meta = extract_category_with_slm("the brakes grind when I stop", ctx)
        assert meta.get("source") == "rules"
    finally:
        reset_process_onnx_embedder()
        reset_slm_cache()


def test_latency_report_uses_measured_time(monkeypatch, reset_ops_db):
    from src.voice.benchmark_latency import TelephonyLatencyBenchmarker

    monkeypatch.setattr(
        TelephonyLatencyBenchmarker,
        "_lognormal_jitter",
        lambda self, median_ms, p95_ms: 10000.0,
    )
    bench = TelephonyLatencyBenchmarker(trials=1)
    result = asyncio.run(bench.benchmark_vad_trigger())
    assert result.samples_ms
    assert max(result.samples_ms) < 1000
    report = asyncio.run(bench.run_full_benchmark())
    assert report["telemetry_source"] == "measured_stage_time"
    assert report["stages"]["total_ttfa"]["p95_ms"] < 1000


def test_latency_policy_failure_is_not_a_pass(monkeypatch):
    from src.voice.wiring import live_turn_budget_verdict

    def broken(*_args, **_kwargs):
        raise RuntimeError("policy missing")

    monkeypatch.setattr("src.voice.policy.latency_verdict", broken)
    verdict = live_turn_budget_verdict(120.0)
    assert verdict["over_budget"] is True
    assert verdict["verdict"] == "unknown"


def test_speech_contracts_do_not_claim_unwired_audio(monkeypatch):
    from src.channels.stt_tts import ServerSpeechStack
    from src.channels.twilio_media import TwilioMediaChannel

    monkeypatch.setenv("STT_API_URL", "http://127.0.0.1:9/stt")
    monkeypatch.setenv("FASTER_WHISPER_MODEL", "tiny")
    speech = ServerSpeechStack()
    assert speech.server_stt_available is False
    assert speech.capabilities()["server_stt"] is False
    assert speech.capabilities()["server_tts"] is False
    assert speech.transcribe_pcm(b"\x00\x02" * 160) is None
    meta = speech.synthesize_meta("hello")
    assert meta["playable"] is False
    assert meta["audio_ref"] is None
    channel = TwilioMediaChannel(speech=speech)
    assert channel.capabilities().server_tts is False


def test_audio_and_transcript_buffers_drop_the_oldest():
    from src.channels.stt_tts import ServerSpeechStack
    from src.channels.twilio_media import TwilioMediaChannel
    from src.voice.telephony_bridge import TelephonyMediaBridge

    speech = ServerSpeechStack()
    for index in range(ServerSpeechStack.PARTIAL_LIMIT + 5):
        speech.transcribe_chunk(f"AUDIO:chunk-{index}")
    assert len(speech._partial) == ServerSpeechStack.PARTIAL_LIMIT
    assert speech._partial[0] == f"AUDIO:chunk-{5}"
    assert speech._partial[-1] == f"AUDIO:chunk-{ServerSpeechStack.PARTIAL_LIMIT + 4}"

    class _WS:
        async def send_text(self, _text: str) -> None:
            return None

    bridge = TelephonyMediaBridge(_WS(), "int_buf", lambda _pcm: None)
    for _ in range(TelephonyMediaBridge.REJECTION_LIMIT + 3):
        asyncio.run(bridge._reject_service_unavailable())
    assert len(bridge.rejections) == TelephonyMediaBridge.REJECTION_LIMIT

    channel = TwilioMediaChannel()

    async def fill() -> None:
        for index in range(TwilioMediaChannel.OUTBOUND_LIMIT + 4):
            await channel.send_turn(f"turn {index}")

    asyncio.run(fill())
    assert len(channel.outbound) == TwilioMediaChannel.OUTBOUND_LIMIT
    assert channel.outbound[0]["text"] == "turn 4"
    assert channel.outbound[-1]["text"] == f"turn {TwilioMediaChannel.OUTBOUND_LIMIT + 3}"
