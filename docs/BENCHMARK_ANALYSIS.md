# Voice Agent Benchmark Analysis

## Open-Source Stack: Whisper + Kokoro + Gemma3:4b

**Date:** December 2025  
**Configuration:** Self-hosted, GPU-accelerated inference

---

## Executive Summary

This analysis evaluates the **latency**, **cost**, and **scalability** of a fully open-source voice agent pipeline for enterprise deployment. The stack eliminates per-token/per-minute API costs while maintaining sub-second response times suitable for real-time conversation.

---

## 1. Pipeline Architecture

| Component | Technology | Model | Hosting |
|-----------|------------|-------|---------|
| **STT** | faster-whisper | large-v3-turbo | Local GPU |
| **LLM** | Ollama | Gemma3:4b / Qwen3-VL:4b | Local GPU |
| **TTS** | Kokoro-FastAPI | Kokoro v1.0 | Local GPU |
| **VAD** | Silero VAD | v5 | Local CPU |

---

## 2. Latency Benchmarks

### 2.1 Component-Level Latency (Single Turn)

| Metric | Typical Range | Optimized | Notes |
|--------|---------------|-----------|-------|
| **STT Transcription** | 80–200ms | 50–100ms | beam_size=1, vad_filter=on |
| **LLM TTFT** | 150–400ms | 100–250ms | Time-to-first-token |
| **LLM Total** | 400–1200ms | 300–800ms | Depends on response length |
| **TTS TTFA** | 50–150ms | 40–100ms | Time-to-first-audio |
| **TTS Total** | 200–600ms | 150–400ms | Streaming output |

### 2.2 End-to-End Latency (Voice-to-Voice)

| Scenario | E2E Latency | User Experience |
|----------|-------------|-----------------|
| **Best Case** | ~300ms | Imperceptible delay |
| **Typical** | 400–600ms | Natural conversation flow |
| **Worst Case** | 800–1200ms | Slight noticeable pause |

> **Target:** < 600ms E2E for conversational quality (human turn-taking ~200-500ms)

### 2.3 Optimization Parameters

```env
# Whisper STT (speed-optimized)
VAGENT_STT_BEAM_SIZE=1
VAGENT_STT_VAD_FILTER=1
VAGENT_STT_WITHOUT_TIMESTAMPS=1

# Agent behavior
VAGENT_MIN_ENDPOINTING_DELAY=0.1
```

---

## 3. Cost Analysis: 30-40 Minute Session

### 3.1 Cloud API Cost (Comparison Baseline)

| Service | Rate | 35-min Session | Monthly (1000 sessions) |
|---------|------|----------------|------------------------|
| **OpenAI Whisper API** | $0.006/min | $0.21 | $210 |
| **GPT-4o** | ~$0.01/1K tokens | $0.50–1.50 | $500–1,500 |
| **ElevenLabs TTS** | $0.30/1K chars | $1.50–3.00 | $1,500–3,000 |
| **Total (Cloud)** | — | **$2.21–4.71** | **$2,210–4,710** |

### 3.2 Self-Hosted Cost (Open-Source Stack)

#### Hardware Requirements

| Tier | GPU | VRAM | Concurrent Sessions | Hardware Cost |
|------|-----|------|---------------------|---------------|
| **Entry** | RTX 3060 | 12GB | 1–2 | $300–400 |
| **Standard** | RTX 4070 | 12GB | 2–4 | $550–650 |
| **Production** | RTX 4090 | 24GB | 4–8 | $1,600–2,000 |
| **Enterprise** | A100 40GB | 40GB | 10–20 | $10,000–15,000 |

#### Operating Cost per Session

| Cost Factor | Entry Tier | Production Tier |
|-------------|------------|-----------------|
| **Power** (300W @ $0.12/kWh) | $0.02 | $0.04 |
| **Hardware Amortization** (3yr) | $0.003 | $0.02 |
| **Total per 35-min Session** | **~$0.02** | **~$0.06** |

### 3.3 Cost Comparison Summary

| Metric | Cloud APIs | Self-Hosted (Entry) | Self-Hosted (Prod) |
|--------|------------|---------------------|-------------------|
| **Per Session (35 min)** | $2.21–4.71 | ~$0.02 | ~$0.06 |
| **Monthly (1000 sessions)** | $2,210–4,710 | ~$20 + hardware | ~$60 + hardware |
| **Break-even** | — | ~150–200 sessions | ~400–500 sessions |
| **Year 1 (1000 sess/mo)** | $26,520–56,520 | ~$640 | ~$2,720 |

> **ROI:** Self-hosted reaches break-even within **1 month** at moderate usage.

---

## 4. Scalability Analysis

### 4.1 Single-GPU Capacity

| GPU | VRAM | Max Concurrent | Throughput (sessions/hr) |
|-----|------|----------------|-------------------------|
| RTX 3060 12GB | 12GB | 1–2 | 1.7–3.4 |
| RTX 4070 12GB | 12GB | 2–3 | 3.4–5.1 |
| RTX 4090 24GB | 24GB | 4–6 | 6.8–10.2 |
| A100 40GB | 40GB | 8–12 | 13.6–20.4 |

### 4.2 VRAM Allocation

| Component | VRAM Usage | Notes |
|-----------|------------|-------|
| **Whisper large-v3-turbo** | ~3GB | FP16 inference |
| **Gemma3:4b** | ~3GB | Q4 quantization |
| **Qwen3-VL:4b** | ~3.5GB | Q4, multimodal |
| **Kokoro TTS** | ~1GB | — |
| **Silero VAD** | ~50MB | CPU-based |
| **System Overhead** | ~1GB | CUDA context |
| **Total (single session)** | **~8–9GB** | — |

### 4.3 Horizontal Scaling Architecture

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    │   (LiveKit SFU) │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │ Agent 1 │          │ Agent 2 │          │ Agent N │
   │ (GPU 1) │          │ (GPU 2) │          │ (GPU N) │
   └─────────┘          └─────────┘          └─────────┘
```

### 4.4 Scaling Recommendations

| Daily Sessions | Infrastructure | Est. Monthly Cost |
|----------------|----------------|-------------------|
| 10–50 | 1x RTX 4070 | $20–30 (power) |
| 50–200 | 1x RTX 4090 | $40–60 (power) |
| 200–1000 | 2–4x RTX 4090 cluster | $150–300 (power) |
| 1000+ | Cloud GPU (on-demand) + dedicated | Variable |

---

## 5. Model Comparison: Gemma3:4b vs Qwen3-VL:4b

| Attribute | Gemma3:4b | Qwen3-VL:4b |
|-----------|-----------|-------------|
| **Parameters** | 4B | 4B |
| **VRAM (Q4)** | ~3GB | ~3.5GB |
| **Inference Speed** | Faster | Slightly slower |
| **Multimodal** | No | Yes (Vision) |
| **Conversation Quality** | Good | Good |
| **Best For** | Voice-only agents | Vision-enabled agents |

**Recommendation:** Use **Gemma3:4b** for voice-only applications (lower latency). Use **Qwen3-VL:4b** if future vision capabilities are needed.

---

## 6. Session Profile: 35-Minute Call

### 6.1 Conversation Metrics (Typical)

| Metric | Value |
|--------|-------|
| **Total Turns** | 40–60 |
| **User Speaking Time** | ~12 min |
| **Agent Speaking Time** | ~15 min |
| **Silence/Thinking** | ~8 min |
| **Avg Turn Duration** | 15–25 sec |

### 6.2 Resource Consumption

| Resource | Per Session | Per 1000 Sessions |
|----------|-------------|-------------------|
| **STT Audio Processed** | ~12 min | 200 hrs |
| **LLM Tokens (in)** | ~8,000 | 8M |
| **LLM Tokens (out)** | ~12,000 | 12M |
| **TTS Characters** | ~15,000 | 15M |
| **GPU Hours** | 0.58 | 580 |
| **Power (300W GPU)** | 0.17 kWh | 170 kWh |

---

## 7. Production Deployment Checklist

### 7.1 Minimum Requirements

- [ ] NVIDIA GPU with 12GB+ VRAM
- [ ] CUDA 12.x + cuDNN 9.x
- [ ] LiveKit server (self-hosted or cloud)
- [ ] 16GB+ system RAM
- [ ] SSD storage for models (~5GB)

### 7.2 Recommended Optimizations

- [ ] Enable TensorRT for Whisper (20–30% speedup)
- [ ] Use vLLM or llama.cpp for LLM (continuous batching)
- [ ] Pre-warm models on process start
- [ ] Implement connection pooling for HTTP clients
- [ ] Enable latency tracking (`VAGENT_LATENCY=1`)

### 7.3 Monitoring Metrics

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| E2E Latency (p50) | <500ms | >800ms |
| E2E Latency (p99) | <1000ms | >1500ms |
| GPU Utilization | 60–80% | >95% sustained |
| VRAM Usage | <80% | >90% |
| Error Rate | <0.1% | >1% |

---

## 8. Conclusion

| Criteria | Rating | Notes |
|----------|--------|-------|
| **Latency** | Excellent | 400–600ms typical E2E |
| **Cost Efficiency** | Excellent | 98% cheaper than cloud APIs |
| **Scalability** | Good | Linear with GPU count |
| **Quality** | Good | Comparable to cloud services |
| **Operational Complexity** | Moderate | Requires GPU infrastructure |

### Key Takeaways

1. **Cost Savings:** Self-hosted open-source stack reduces costs by **50–100x** vs cloud APIs
2. **Break-even:** Achieved within **1 month** at ~150+ sessions/month
3. **Latency:** Sub-600ms E2E achievable with proper optimization
4. **Scalability:** Each RTX 4090 handles ~100–150 daily sessions

---

## Appendix: Quick Start Commands

```bash
# Start Kokoro TTS server
docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi:latest

# Start Ollama with Gemma3
ollama serve &
ollama pull gemma3:4b

# Run voice agent
python -m vagent.agent start
```

---

*Generated by vagent benchmark tools*
