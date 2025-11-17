# content_creation
The system employs a sophisticated 5-agent orchestration framework built on LangChain/LangGraph, where specialized AI agents autonomously handle content discovery, multi-source fact-checking, creative scriptwriting, quality review, and audio synthesis.

📋 Table of Contents
Features
Architecture
Demo
Installation
Configuration
Usage
Agent Workflow
Output
Cost Analysis
Customization
Troubleshooting
Contributing
License
✨ Features🤖 Multi-Agent Orchestration

5 Specialized AI Agents working autonomously
Smart Routing with conditional logic and revision loops
State Management tracking full workflow history
🔄 Flexible LLM Integration

5+ LLM Providers supported: OpenAI GPT-4, XAI Grok-4, Anthropic Claude, Google Gemini, Groq
Hot-swappable models with unified interface
Cost optimization through provider switching
✅ Automated Quality Control

95%+ factual accuracy through multi-source verification
Iterative refinement with up to 3 revision cycles
Engagement optimization following viral content structure
🎙️ Local GPU Acceleration

VibeVoice TTS for voice cloning and synthesis
Wan2.2 for audio-to-video animation
LTX Video for scene generation
NVIDIA CUDA powered processing
📊 Production-Grade Features

Comprehensive logging with JSON export
Duration tracking for performance optimization
Audio-ready scripts with automated cleaning
Multiple format support (YouTube Shorts, TikTok, Instagram Reels)

┌─────────────────────────────────────────────────────────────┐
│                    User Input / Config                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │   Agent 1: Trend Finder │
          │  Discovers 10 topics    │
          │  Scores by popularity   │
          └────────┬───────────────┘
                   │
                   ▼
          ┌────────────────────────┐
          │ Agent 2: Researcher     │
          │ Fact-checks topics      │
          │ Recalibrates scores     │
          └────────┬───────────────┘
                   │
                   ▼
          ┌────────────────────────┐
          │   Human Selection       │
          │  Choose from Top 5      │
          └────────┬───────────────┘
                   │
                   ▼
          ┌────────────────────────┐
          │ Agent 3: Scriptwriter   │
          │ Creates viral scripts   │
          │ 7-part structure        │
          └────────┬───────────────┘
                   │
                   ▼
          ┌────────────────────────┐
          │ Agent 4: Reviewer       │
          │ Validates quality       │
          │ Requests revisions      │
          └────────┬───────────────┘
                   │
            ┌──────┴──────┐
            │             │
        [Approved]   [Needs Revision]
            │             │
            │             └───────┐
            │                     │
            ▼                     ▼
   ┌────────────────┐    ┌───────────────┐
   │ Agent 5: Audio │    │ Agent 3: Revise│
   │ TTS Generation │    │ (Max 3 cycles) │
   │ Video Synthesis│    └───────┬────────┘
   └────────────────┘            │
            │                    │
            │                    ▼
            │           ┌─────────────────┐
            │           │ Agent 4: Review │
            │           │    (Again)      │
            │           └────────┬────────┘
            │                    │
            └────────────────────┘
                       │
                       ▼
            ┌──────────────────┐
            │  Final Output:   │
            │  • Scripts (.txt)│
            │  • Audio (.mp3)  │
            │  • Logs (.json)  │
            └──────────────────┘



            
