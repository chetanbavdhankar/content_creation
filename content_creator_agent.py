"""
Agentic Content Creator System using LangChain/LangGraph
Supports GPT-4 and other LLM models
"""

import os
from typing import TypedDict, Annotated, List, Dict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import operator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify API key is loaded (optional - for debugging)
def verify_api_key():
    """Check if the required API key is set"""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "xai": "XAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY"
    }
    
    required_key = key_map.get(LLM_PROVIDER)
    if required_key and not os.getenv(required_key):
        raise ValueError(f"❌ {required_key} not found in .env file or environment variables!")
    print(f"✅ API key for {LLM_PROVIDER.upper()} loaded successfully")

# Configuration - Switch between models easily
LLM_PROVIDER = "xai"  # Options: "openai", "anthropic", "xai", "google", "groq"
MODEL_NAME = "grok-4-0709"  # See model options below

"""
Supported Models by Provider:
- openai: "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"
- anthropic: "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"
- xai: "grok-4-0709", "grok-2-1212", "grok-2-vision-1212", "grok-beta"
- google: "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"
- groq: "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"
"""

# Initialize LLM based on provider
def get_llm(temperature=0.7):
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(
            model=MODEL_NAME,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif LLM_PROVIDER == "anthropic":
        return ChatAnthropic(
            model=MODEL_NAME,
            temperature=temperature,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    elif LLM_PROVIDER == "xai":
        # X.AI (Grok) uses OpenAI-compatible API
        return ChatOpenAI(
            model=MODEL_NAME,
            temperature=temperature,
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
    elif LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    elif LLM_PROVIDER == "groq":
        # Groq uses OpenAI-compatible API
        return ChatOpenAI(
            model=MODEL_NAME,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")


# Define the state structure
class AgentState(TypedDict):
    trending_topics: List[Dict]  # From Agent 1
    researched_topics: List[Dict]  # From Agent 2
    top_5_topics: List[Dict]  # Filtered top 5
    selected_topics: List[str]  # Human-selected topics
    scripts: Dict[str, str]  # Topic -> Script mapping
    reviewed_scripts: Dict[str, Dict]  # Topic -> {approved: bool, feedback: str}
    agent_messages: Annotated[List[str], operator.add]  # Log of agent actions
    current_iteration: int  # For script revision tracking
    interaction_log: Dict  # Complete interaction history


def log_interaction(state: AgentState, agent_name: str, action: str, data: Dict, duration: float = None):
    """Log agent interactions in a structured format"""
    from datetime import datetime
    
    # Ensure interaction_log exists and is properly initialized
    if "interaction_log" not in state or not isinstance(state.get("interaction_log"), dict):
        state["interaction_log"] = {
            "session_start": datetime.now().isoformat(),
            "llm_provider": LLM_PROVIDER,
            "model": MODEL_NAME,
            "agents": {}
        }
    
    # Ensure agents key exists
    if "agents" not in state["interaction_log"]:
        state["interaction_log"]["agents"] = {}
    
    if agent_name not in state["interaction_log"]["agents"]:
        state["interaction_log"]["agents"][agent_name] = {
            "executions": [],
            "total_calls": 0
        }
    
    interaction_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "data": data,
        "duration_seconds": duration
    }
    
    state["interaction_log"]["agents"][agent_name]["executions"].append(interaction_entry)
    state["interaction_log"]["agents"][agent_name]["total_calls"] += 1
    
    return state


# Agent 1: Find Trending Historical Topics
def agent_1_find_trends(state: AgentState) -> AgentState:
    """Find top 10 trending historical topics"""
    import time
    start_time = time.time()
    
    print("\n🔍 AGENT 1: Finding trending historical topics...")
    
    llm = get_llm(temperature=0.8)
    
    prompt = """You are a content research specialist. Find 10 trending topics related to OLD HISTORY 
    that are currently popular in news, social media, or have recent relevance.
    
    For each topic, provide:
    1. A simple, catchy title (max 10 words)
    2. Brief information (2-3 sentences)
    3. A score out of 10 for:
       - Popularity (how trending it is)
       - Video-worthiness (how suitable for short video content)
    
    Calculate total score = (Popularity + Video-worthiness) / 2
    
    Format your response as a Python list of dictionaries:
    [
        {
            "title": "Title here",
            "info": "Brief info here",
            "popularity": 8,
            "video_worthiness": 9,
            "total_score": 8
        },
        ...
    ]
    
    Order them by total_score (highest first). Return ONLY the Python list, no other text."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Parse the response (basic parsing - in production, use more robust parsing)
    import ast
    try:
        topics = ast.literal_eval(response.content.strip())
    except:
        # Fallback: extract text between brackets
        content = response.content
        start = content.find('[')
        end = content.rfind(']') + 1
        topics = ast.literal_eval(content[start:end])
    
    duration = time.time() - start_time
    
    # Log interaction
    log_interaction(state, "Agent_1_Trend_Finder", "find_trending_topics", {
        "prompt": prompt,
        "response_raw": response.content,
        "topics_found": len(topics),
        "topics": topics
    }, duration)
    
    state["trending_topics"] = topics
    state["agent_messages"].append(f"Agent 1: Found {len(topics)} trending historical topics")
    
    print(f"✅ Agent 1: Found {len(topics)} topics (took {duration:.2f}s)")
    return state


# Agent 2: Research and Verify Topics
def agent_2_research(state: AgentState) -> AgentState:
    """Research and verify each topic, recalibrate scores"""
    import time
    start_time = time.time()
    
    print("\n🔬 AGENT 2: Researching and verifying topics...")
    
    llm = get_llm(temperature=0.3)  # Lower temperature for factual accuracy
    
    researched = []
    research_details = []
    
    for idx, topic in enumerate(state["trending_topics"], 1):
        topic_start = time.time()
        print(f"  Researching topic {idx}/10: {topic['title']}")
        
        prompt = f"""You are a fact-checker and historical researcher. Verify the following historical topic:

Title: {topic['title']}
Info: {topic['info']}

Tasks:
1. Verify if this topic is FACTUAL and ACCURATE
2. Check if there are any historical inaccuracies
3. Assess its suitability for short-form video content
4. Provide 2-3 credible reference sources or mention general historical consensus

Based on your research, recalibrate the score (0-10 integer only) considering:
- Factual accuracy (weight: 40%)
- Content richness for storytelling (weight: 30%)
- Visual/engagement potential (weight: 30%)

Respond in this exact format:
VERIFIED: Yes/No
ACCURACY_NOTES: [Your brief notes]
REFERENCES: [List 2-3 sources or "General historical consensus"]
RECALIBRATED_SCORE: [0-10 integer]
REASONING: [Brief explanation]"""

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        
        # Parse response
        verified = "Yes" in content.split("VERIFIED:")[1].split("\n")[0]
        
        # Extract score
        try:
            score_line = [line for line in content.split("\n") if "RECALIBRATED_SCORE" in line][0]
            new_score = int(''.join(filter(str.isdigit, score_line.split(":")[1])))
        except:
            new_score = topic["total_score"]  # Keep original if parsing fails
        
        topic_duration = time.time() - topic_start
        
        researched_topic = {
            **topic,
            "verified": verified,
            "research_notes": content,
            "recalibrated_score": new_score
        }
        researched.append(researched_topic)
        
        # Log each research interaction
        research_details.append({
            "topic_title": topic['title'],
            "original_score": topic['total_score'],
            "new_score": new_score,
            "verified": verified,
            "prompt": prompt,
            "response": content,
            "duration_seconds": topic_duration
        })
    
    # Sort by recalibrated score
    researched.sort(key=lambda x: x["recalibrated_score"], reverse=True)
    
    duration = time.time() - start_time
    
    # Log interaction
    log_interaction(state, "Agent_2_Researcher", "research_and_verify", {
        "topics_researched": len(researched),
        "research_details": research_details,
        "top_5_after_research": researched[:5]
    }, duration)
    
    state["researched_topics"] = researched
    state["top_5_topics"] = researched[:5]
    state["agent_messages"].append("Agent 2: Researched and verified all topics")
    
    print(f"✅ Agent 2: Completed research and verification (took {duration:.2f}s)")
    return state


# Display Top 5 for Human Selection
def display_top_5(state: AgentState) -> AgentState:
    """Display top 5 topics for human selection"""
    import time
    start_time = time.time()
    
    print("\n" + "="*80)
    print("📊 TOP 5 RECOMMENDED TOPICS FOR CONTENT CREATION")
    print("="*80)
    
    for idx, topic in enumerate(state["top_5_topics"], 1):
        print(f"\n{idx}. {topic['title']}")
        print(f"   Score: {topic['recalibrated_score']}/10")
        print(f"   Verified: {'✓' if topic['verified'] else '✗'}")
        print(f"   Info: {topic['info']}")
        print(f"   {'-'*70}")
    
    print("\n" + "="*80)
    
    # Get human input
    print("\nSelect topics for script creation (comma-separated numbers, e.g., 1,3,5):")
    selection = input("Your selection: ").strip()
    
    selected_indices = [int(x.strip()) - 1 for x in selection.split(",")]
    selected_topics = [state["top_5_topics"][i]["title"] for i in selected_indices if 0 <= i < 5]
    
    duration = time.time() - start_time
    
    # Log human interaction
    log_interaction(state, "Human_Selection", "select_topics", {
        "top_5_presented": [t['title'] for t in state["top_5_topics"]],
        "user_input": selection,
        "selected_indices": selected_indices,
        "selected_topics": selected_topics,
        "selection_count": len(selected_topics)
    }, duration)
    
    state["selected_topics"] = selected_topics
    state["agent_messages"].append(f"Human selected {len(selected_topics)} topics")
    
    return state


# Agent 3: Create Scripts
def agent_3_create_scripts(state: AgentState) -> AgentState:
    """Create short-form video scripts for selected topics"""
    import time
    start_time = time.time()
    
    print("\n✍️ AGENT 3: Creating scripts for selected topics...")
    
    llm = get_llm(temperature=0.9)  # Higher temperature for creativity
    
    scripts = {}
    script_details = []
    
    for topic_title in state["selected_topics"]:
        topic_start = time.time()
        # Find the full topic info
        topic = next((t for t in state["top_5_topics"] if t["title"] == topic_title), None)
        if not topic:
            continue
        
        print(f"  Creating script for: {topic_title}")
        
        prompt = f"""You are a viral short-form video scriptwriter specializing in historical content.
Create an engaging script for a 15-20 second video about this topic:

Title: {topic['title']}
Info: {topic['info']}

Your script MUST follow this exact structure:

1. INSTANT HOOK (1 second)
   - One punchy line that stops the scroll
   
2. CURIOSITY TRIGGER
   - Suggest something surprising is about to be revealed
   
3. QUICK SETUP
   - Fast introduction, tight and punchy
   
4. MAIN INSIGHT 
   - Core information in rapid, digestible lines
   
5. TWIST OR UNEXPECTED ANGLE 
   - Surprising detail that reframes the topic
   
6. MODERN/PERSONAL RELEVANCE 
   - Connect to present day or current situation
   
7. HIGH-ENGAGEMENT CTA 
   - Simple question for comments

Format each section clearly with headers. Make it ENGAGING, FACTUAL, and CURIOSITY-DRIVEN.
Use conversational language. Avoid clichés. Be specific with facts and details. 
Dont refer to any social media platform in the script."""

        response = llm.invoke([HumanMessage(content=prompt)])
        scripts[topic_title] = response.content
        
        topic_duration = time.time() - topic_start
        
        script_details.append({
            "topic_title": topic_title,
            "topic_score": topic.get('recalibrated_score', 0),
            "prompt": prompt,
            "script": response.content,
            "script_length": len(response.content),
            "duration_seconds": topic_duration
        })
    
    duration = time.time() - start_time
    
    # Log interaction
    log_interaction(state, "Agent_3_Scriptwriter", "create_scripts", {
        "scripts_created": len(scripts),
        "script_details": script_details,
        "iteration": state.get("current_iteration", 0)
    }, duration)
    
    state["scripts"] = scripts
    state["agent_messages"].append(f"Agent 3: Created scripts for {len(scripts)} topics")
    
    print(f"✅ Agent 3: Created {len(scripts)} scripts (took {duration:.2f}s)")
    return state


# Agent 4: Review Scripts
def agent_4_review_scripts(state: AgentState) -> AgentState:
    """Review scripts for logic and correctness"""
    import time
    start_time = time.time()
    
    print("\n🔍 AGENT 4: Reviewing scripts...")
    
    llm = get_llm(temperature=0.2)  # Low temperature for critical review
    
    # Initialize reviewed dict with existing approved scripts if this is a revision cycle
    if state.get("current_iteration", 0) > 0 and state.get("reviewed_scripts"):
        reviewed = state["reviewed_scripts"].copy()
    else:
        reviewed = {}
    
    needs_revision = []
    review_details = []
    
    for topic_title, script in state["scripts"].items():
        # Skip already approved scripts in revision cycles
        if state.get("current_iteration", 0) > 0:
            if topic_title in reviewed and reviewed[topic_title].get("approved", False):
                print(f"  ✓ Skipping already approved script: {topic_title}")
                continue
        
        topic_start = time.time()
        # Find topic info
        topic = next((t for t in state["top_5_topics"] if t["title"] == topic_title), None)
        
        print(f"  Reviewing script for: {topic_title}")
        
        prompt = f"""You are a critical script reviewer for historical content. Review this script:

TOPIC: {topic_title}
TOPIC INFO: {topic['info'] if topic else 'N/A'}

SCRIPT:
{script}

Evaluate:
1. FACTUAL ACCURACY: Are all historical facts correct?
2. LOGICAL FLOW: Does the structure make sense?
3. ENGAGEMENT: Is it compelling for short-form video?
4. SCRIPT STRUCTURE: Does it follow the 7-part format correctly?
5. TIMING: Are sections appropriately timed?

Respond in this format:
APPROVED: Yes/No
ISSUES: [List any problems found, or "None" if approved]
FACTUAL_ERRORS: [Any historical inaccuracies]
STRUCTURAL_PROBLEMS: [Any flow or format issues]
FEEDBACK: [Specific suggestions for improvement if not approved]"""

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        
        approved = "APPROVED: Yes" in content
        
        topic_duration = time.time() - topic_start
        
        reviewed[topic_title] = {
            "approved": approved,
            "feedback": content
        }
        
        if not approved:
            needs_revision.append(topic_title)
        
        review_details.append({
            "topic_title": topic_title,
            "approved": approved,
            "prompt": prompt,
            "feedback": content,
            "duration_seconds": topic_duration
        })
    
    duration = time.time() - start_time
    
    # Log interaction
    log_interaction(state, "Agent_4_Reviewer", "review_scripts", {
        "scripts_reviewed": len(review_details),
        "approved_count": sum(1 for r in reviewed.values() if r["approved"]),
        "needs_revision_count": len(needs_revision),
        "needs_revision_topics": needs_revision,
        "review_details": review_details,
        "iteration": state.get("current_iteration", 0)
    }, duration)
    
    state["reviewed_scripts"] = reviewed
    state["agent_messages"].append(f"Agent 4: Reviewed {len(review_details)} scripts")
    
    # Display results
    print("\n" + "="*80)
    print("📋 SCRIPT REVIEW RESULTS")
    print("="*80)
    
    for topic_title, review in reviewed.items():
        status = "✅ APPROVED" if review["approved"] else "❌ NEEDS REVISION"
        print(f"\n{status}: {topic_title}")
        if not review["approved"]:
            print(f"\nFeedback:\n{review['feedback']}")
    
    # Handle revisions if needed
    if needs_revision and state.get("current_iteration", 0) < 3:
        print(f"\n⚠️ {len(needs_revision)} script(s) need revision. Requesting Agent 3 to revise...")
        state["current_iteration"] = state.get("current_iteration", 0) + 1
    
    print(f"✅ Agent 4: Completed review (took {duration:.2f}s)")
    return state


# Router function to decide if revision is needed
def should_revise_scripts(state: AgentState) -> Literal["revise", "end"]:
    """Determine if scripts need revision"""
    if not state.get("reviewed_scripts"):
        return "end"
    
    needs_revision = any(not review["approved"] for review in state["reviewed_scripts"].values())
    max_iterations = 3
    
    if needs_revision and state.get("current_iteration", 0) < max_iterations:
        return "revise"
    return "end"


# Agent 3 Revision
def agent_3_revise_scripts(state: AgentState) -> AgentState:
    """Revise scripts based on Agent 4 feedback"""
    import time
    start_time = time.time()
    
    print("\n✍️ AGENT 3: Revising scripts based on feedback...")
    
    llm = get_llm(temperature=0.9)
    revision_details = []
    
    for topic_title, review in state["reviewed_scripts"].items():
        if not review["approved"]:
            topic_start = time.time()
            print(f"  Revising script for: {topic_title}")
            
            topic = next((t for t in state["top_5_topics"] if t["title"] == topic_title), None)
            old_script = state["scripts"][topic_title]
            
            prompt = f"""You are revising a short-form video script based on reviewer feedback.

TOPIC: {topic_title}
TOPIC INFO: {topic['info'] if topic else 'N/A'}

PREVIOUS SCRIPT:
{old_script}

REVIEWER FEEDBACK:
{review['feedback']}

Create an IMPROVED script that addresses all the feedback. Follow the same 7-part structure:
1. Instant Hook (1s)
2. Curiosity Trigger (3-4s)
3. Quick Setup (6-7s)
4. Main Insight (8-10s)
5. Twist/Unexpected Angle (8-10s)
6. Modern Relevance (5-6s)
7. High-Engagement CTA (3-4s)

Ensure factual accuracy and logical flow."""

            response = llm.invoke([HumanMessage(content=prompt)])
            state["scripts"][topic_title] = response.content
            
            topic_duration = time.time() - topic_start
            
            revision_details.append({
                "topic_title": topic_title,
                "old_script": old_script,
                "new_script": response.content,
                "reviewer_feedback": review['feedback'],
                "prompt": prompt,
                "duration_seconds": topic_duration
            })
    
    duration = time.time() - start_time
    
    # Log interaction
    log_interaction(state, "Agent_3_Scriptwriter", "revise_scripts", {
        "iteration": state['current_iteration'],
        "scripts_revised": len(revision_details),
        "revision_details": revision_details
    }, duration)
    
    state["agent_messages"].append(f"Agent 3: Revised scripts (Iteration {state['current_iteration']})")
    print(f"✅ Agent 3: Completed revisions (took {duration:.2f}s)")
    return state


# Build the LangGraph workflow
def build_content_creator_graph():
    """Build the agentic workflow graph"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent_1", agent_1_find_trends)
    workflow.add_node("agent_2", agent_2_research)
    workflow.add_node("display_top_5", display_top_5)
    workflow.add_node("agent_3", agent_3_create_scripts)
    workflow.add_node("agent_4", agent_4_review_scripts)
    workflow.add_node("agent_3_revise", agent_3_revise_scripts)
    
    # Define the flow
    workflow.set_entry_point("agent_1")
    workflow.add_edge("agent_1", "agent_2")
    workflow.add_edge("agent_2", "display_top_5")
    workflow.add_edge("display_top_5", "agent_3")
    workflow.add_edge("agent_3", "agent_4")
    
    # Conditional edge for revisions
    workflow.add_conditional_edges(
        "agent_4",
        should_revise_scripts,
        {
            "revise": "agent_3_revise",
            "end": END
        }
    )
    workflow.add_edge("agent_3_revise", "agent_4")
    
    return workflow.compile()


def clean_script_for_audio(script):
    """Remove section headers and structure markers, keep only the actual script content"""
    import re
    
    lines = script.split('\n')
    cleaned_lines = []
    
    # Patterns to identify section headers/markers to remove
    section_patterns = [
        r'^\d+\.\s*(INSTANT HOOK|CURIOSITY TRIGGER|QUICK SETUP|MAIN INSIGHT|TWIST|UNEXPECTED ANGLE|MODERN|PERSONAL|RELEVANCE|HIGH-ENGAGEMENT|CTA|CALL TO ACTION)',
        r'^(INSTANT HOOK|CURIOSITY TRIGGER|QUICK SETUP|MAIN INSIGHT|TWIST|UNEXPECTED ANGLE|MODERN|PERSONAL|RELEVANCE|HIGH-ENGAGEMENT|CTA|CALL TO ACTION)',
        r'^\(\d+[\s\-]*(second|seconds|s)\)',
        r'^[\-\*]+\s*$',  # Lines with just dashes or asterisks
        r'^#{1,6}\s+',  # Markdown headers
    ]
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Check if line matches any section pattern
        is_section_marker = False
        for pattern in section_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                is_section_marker = True
                break
        
        # Skip section markers, keep actual content
        if not is_section_marker:
            # Remove any leading markers like "- " or "* "
            line = re.sub(r'^[\-\*]\s+', '', line)
            if line:  # Only add non-empty lines
                cleaned_lines.append(line)
    
    # Join with double line breaks for better readability
    return '\n\n'.join(cleaned_lines)


def save_scripts_to_file(approved_scripts, state):
    """Save all approved scripts to a single text file - optimized for audio transcription"""
    from datetime import datetime
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scripts/content_scripts_{timestamp}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # Write header
            f.write("="*80 + "\n")
            f.write("CONTENT CREATOR - APPROVED VIDEO SCRIPTS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"LLM Provider: {LLM_PROVIDER.upper()}\n")
            f.write(f"Model: {MODEL_NAME}\n")
            f.write(f"Total Scripts: {len(approved_scripts)}\n")
            f.write("="*80 + "\n\n")
            
            # Write each script
            for idx, (topic_title, script) in enumerate(approved_scripts.items(), 1):
                f.write("\n" + "="*80 + "\n")
                f.write(f"SCRIPT #{idx}\n")
                f.write("="*80 + "\n")
                f.write(f"TOPIC: {topic_title}\n")
                f.write("-"*80 + "\n")
                
                # Find and add topic info
                topic = next((t for t in state["top_5_topics"] if t["title"] == topic_title), None)
                if topic:
                    f.write(f"Score: {topic['recalibrated_score']}/10\n")
                    f.write(f"Brief Info: {topic['info']}\n")
                    f.write("-"*80 + "\n")
                
                f.write("\n--- AUDIO-READY SCRIPT (Read line by line) ---\n\n")
                
                # Clean the script for audio transcription
                cleaned_script = clean_script_for_audio(script)
                f.write(cleaned_script)
                f.write("\n\n")
            
            # Write footer
            f.write("\n" + "="*80 + "\n")
            f.write("END OF SCRIPTS\n")
            f.write("="*80 + "\n")
        
        print(f"\n💾 Scripts saved successfully to: {filename}")
        print(f"📍 File location: {os.path.abspath(filename)}")
        
    except Exception as e:
        print(f"\n❌ Error saving scripts to file: {e}")
        print("Scripts are still displayed above.")


def save_interaction_log(state):
    """Save complete interaction log to JSON file"""
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scripts/interaction_log_{timestamp}.json"
    
    try:
        # Ensure interaction_log exists
        if "interaction_log" not in state or not state["interaction_log"]:
            state["interaction_log"] = {
                "agents": {}
            }
        
        # Add/update session metadata
        if "session_start" not in state["interaction_log"]:
            state["interaction_log"]["session_start"] = datetime.now().isoformat()
        
        if "llm_provider" not in state["interaction_log"]:
            state["interaction_log"]["llm_provider"] = LLM_PROVIDER
        
        if "model" not in state["interaction_log"]:
            state["interaction_log"]["model"] = MODEL_NAME
        
        # Add session end time
        state["interaction_log"]["session_end"] = datetime.now().isoformat()
        
        # Calculate total duration
        if "session_start" in state["interaction_log"]:
            try:
                start = datetime.fromisoformat(state["interaction_log"]["session_start"])
                end = datetime.fromisoformat(state["interaction_log"]["session_end"])
                state["interaction_log"]["total_duration_seconds"] = (end - start).total_seconds()
            except:
                state["interaction_log"]["total_duration_seconds"] = 0
        
        # Save to JSON file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(state["interaction_log"], f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Interaction log saved to: {filename}")
        print(f"📍 File location: {os.path.abspath(filename)}")
        
        return filename
        
    except Exception as e:
        print(f"\n❌ Error saving interaction log: {e}")
        return None


def print_interaction_summary(state):
    """Print a summary of all agent interactions"""
    if "interaction_log" not in state or not state["interaction_log"]:
        print("\n⚠️ No interaction log available")
        return
    
    print("\n" + "="*80)
    print("📊 AGENT INTERACTION SUMMARY")
    print("="*80)
    
    log = state["interaction_log"]
    
    # Get duration with fallback
    duration = log.get('total_duration_seconds', 0)
    if duration == 0 and "session_start" in log and "session_end" in log:
        from datetime import datetime
        try:
            start = datetime.fromisoformat(log["session_start"])
            end = datetime.fromisoformat(log["session_end"])
            duration = (end - start).total_seconds()
        except:
            duration = 0
    
    print(f"\n🕐 Session Duration: {duration:.2f} seconds")
    print(f"🤖 LLM Provider: {log.get('llm_provider', LLM_PROVIDER).upper()}")
    print(f"🧠 Model: {log.get('model', MODEL_NAME)}")
    
    if "agents" in log and log["agents"]:
        print(f"\n👥 Total Agent Actions: {len(log['agents'])}")
        
        for agent_name, agent_data in log["agents"].items():
            print(f"\n  📌 {agent_name}")
            print(f"     Total Calls: {agent_data.get('total_calls', 0)}")
            
            total_duration = sum(
                exec.get('duration_seconds', 0) 
                for exec in agent_data.get('executions', [])
                if exec.get('duration_seconds')
            )
            print(f"     Total Time: {total_duration:.2f}s")
            
            for idx, execution in enumerate(agent_data.get('executions', []), 1):
                action = execution.get('action', 'N/A')
                exec_duration = execution.get('duration_seconds', 0)
                print(f"       └─ Execution {idx}: {action} ({exec_duration:.2f}s)")
    else:
        print("\n⚠️ No agent execution data available")
    
    print("\n" + "="*80)


# Main execution
def main():
    """Run the content creator agent system"""
    print("🚀 Starting Agentic Content Creator System")
    print(f"📡 Using: {LLM_PROVIDER.upper()} - {MODEL_NAME}")
    print("="*80)
    
    # Verify API key is loaded
    verify_api_key()
    
    # Initialize state
    initial_state = {
        "trending_topics": [],
        "researched_topics": [],
        "top_5_topics": [],
        "selected_topics": [],
        "scripts": {},
        "reviewed_scripts": {},
        "agent_messages": [],
        "current_iteration": 0,
        "interaction_log": {
            "agents": {}
        }
    }
    
    # Build and run the graph
    app = build_content_creator_graph()
    
    # Execute the workflow
    final_state = app.invoke(initial_state)
    
    # Display final results
    print("\n" + "="*80)
    print("🎉 FINAL RESULTS")
    print("="*80)
    
    print("\n📝 APPROVED SCRIPTS:")
    approved_scripts = {}
    for topic_title, script in final_state["scripts"].items():
        if final_state["reviewed_scripts"][topic_title]["approved"]:
            print(f"\n{'='*80}")
            print(f"TOPIC: {topic_title}")
            print(f"{'='*80}")
            print(script)
            approved_scripts[topic_title] = script
    
    # Save scripts to file
    save_scripts_to_file(approved_scripts, final_state)
    
    # Print interaction summary
    print_interaction_summary(final_state)
    
    # Save complete interaction log
    log_file = save_interaction_log(final_state)
    
    print("\n" + "="*80)
    print("✅ Workflow Complete!")
    if log_file:
        print(f"📊 Access full interaction log: {log_file}")
    print("="*80)
    
    return final_state


if __name__ == "__main__":
    # API keys are now automatically loaded from .env file!
    # Just create a .env file in the same folder with your keys:
    # 
    # Example .env file:
    # OPENAI_API_KEY=sk-proj-xxxxx
    # GROQ_API_KEY=gsk-xxxxx
    # GOOGLE_API_KEY=AIza-xxxxx
    # XAI_API_KEY=xai-xxxxx
    # ANTHROPIC_API_KEY=sk-ant-xxxxx
    
    main()