"""
Raindrop SDK Test - Compare with Sentrial
==========================================

This example tests Raindrop's features:
- Track AI interactions
- Identify users  
- Track signals (thumbs up/down)
- Partial event tracking (interactions)
- Tracing with decorators

To run:
1. pip install raindrop-ai google-generativeai
2. Set RAINDROP_WRITE_KEY in your environment
3. Set GEMINI_API_KEY in your environment
4. python raindrop_test.py

Get your Raindrop write key at: https://app.raindrop.ai
"""

import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# RAINDROP SETUP
# ============================================================
try:
    import raindrop.analytics as raindrop
    RAINDROP_AVAILABLE = True
except ImportError:
    print("⚠️  Raindrop SDK not installed. Run: pip install raindrop-ai")
    RAINDROP_AVAILABLE = False

# ============================================================
# GEMINI SETUP (for actual AI responses)
# ============================================================
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
        print("⚠️  GEMINI_API_KEY not set")
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai not installed")


def test_basic_tracking():
    """Test basic AI interaction tracking"""
    print("\n" + "="*60)
    print("TEST 1: Basic AI Tracking")
    print("="*60)
    
    if not RAINDROP_AVAILABLE:
        print("❌ Skipping - Raindrop not available")
        return
    
    # Initialize Raindrop
    write_key = os.getenv("RAINDROP_WRITE_KEY")
    if not write_key:
        print("❌ RAINDROP_WRITE_KEY not set")
        return
    
    raindrop.init(write_key)
    raindrop.set_debug_logs(True)
    
    # Identify user
    raindrop.identify(
        user_id="test_user_001",
        traits={
            "name": "Test User",
            "email": "test@example.com",
            "plan": "trial"
        }
    )
    print("✅ User identified")
    
    # Track a simple AI interaction
    user_input = "What is the capital of France?"
    
    if GEMINI_AVAILABLE:
        # Configure a friendly system prompt for the support chatbot
    SYSTEM_PROMPT = """You are a warm, empathetic, and highly knowledgeable support assistant. Your goal is to be genuinely helpful while maintaining a friendly, conversational tone. 

    Key traits:
    - Always respond with warmth and understanding
    - Show genuine care for the user's needs and concerns
    - Use encouraging language and positive framing
    - Be patient and never make users feel rushed or judged
    - Acknowledge when users might be frustrated and respond with extra empathy
    - Provide clear, actionable advice while being supportive
    - Use a conversational, human-like tone rather than robotic responses

    Remember: Every interaction is an opportunity to make someone's day a little better while solving their problem effectively."""
    
    model = genai.GenerativeModel(
        "gemini-2.5-pro",
        system_instruction=SYSTEM_PROMPT
    )
        response = model.generate_content(user_input)
        ai_output = response.text
    else:
        ai_output = "The capital of France is Paris."
    
    raindrop.track_ai(
        user_id="test_user_001",
        event="user_message",
        model="gemini-2.5-pro",
        input=user_input,
        output=ai_output,
        convo_id="test_convo_001",
        properties={
            "system_prompt": "You are a helpful assistant.",
            "experiment": "raindrop_test"
        }
    )
    print(f"✅ Tracked AI interaction")
    print(f"   Input: {user_input}")
    print(f"   Output: {ai_output[:100]}...")
    
    raindrop.flush()
    print("✅ Events flushed")


def test_signals():
    """Test signal tracking (thumbs up/down)"""
    print("\n" + "="*60)
    print("TEST 2: Signal Tracking")
    print("="*60)
    
    if not RAINDROP_AVAILABLE:
        print("❌ Skipping - Raindrop not available")
        return
    
    # First track an event to get an event_id
    event_id = f"evt_{int(time.time())}"
    
    raindrop.track_ai(
        user_id="test_user_001",
        event="chatbot_response",
        event_id=event_id,
        model="gemini-2.5-pro",
        input="How do I reset my password?",
        output="You can reset your password by clicking 'Forgot Password' on the login page.",
    )
    print(f"✅ Created event: {event_id}")
    
    # Track thumbs up signal
    raindrop.track_signal(
        event_id=event_id,
        name="thumbs_up",
        signal_type="default",
        sentiment="POSITIVE"
    )
    print("✅ Tracked thumbs_up signal")
    
    # Track feedback signal
    raindrop.track_signal(
        event_id=event_id,
        name="user_feedback",
        signal_type="feedback",
        comment="This was very helpful, thanks!"
    )
    print("✅ Tracked feedback signal")
    
    raindrop.flush()


def test_partial_events():
    """Test partial event tracking with begin/finish"""
    print("\n" + "="*60)
    print("TEST 3: Partial Event Tracking (Interactions)")
    print("="*60)
    
    if not RAINDROP_AVAILABLE:
        print("❌ Skipping - Raindrop not available")
        return
    
    # Start an interaction
    interaction = raindrop.begin(
        user_id="test_user_001",
        event="code_generation",
        input="Write a Python function to calculate fibonacci"
    )
    print(f"✅ Started interaction: {interaction.id}")
    
    # Simulate user adding more context
    time.sleep(0.5)
    interaction.add_attachments([
        {
            "type": "text",
            "name": "Additional context",
            "value": "It should be recursive and handle edge cases",
            "role": "input"
        }
    ])
    print("✅ Added attachment")
    
    # Simulate AI generating output
    time.sleep(0.5)
    
    code_output = '''def fibonacci(n):
    """Calculate the nth Fibonacci number recursively."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
    
    interaction.finish(
        output=code_output,
        attachments=[
            {
                "type": "code",
                "name": "fibonacci.py",
                "value": code_output,
                "role": "output",
                "language": "python"
            }
        ]
    )
    print("✅ Finished interaction with code output")
    
    raindrop.flush()


def test_complex_conversation():
    """Test a multi-turn conversation with actual Gemini"""
    print("\n" + "="*60)
    print("TEST 4: Complex Multi-Turn Conversation")
    print("="*60)
    
    if not RAINDROP_AVAILABLE or not GEMINI_AVAILABLE:
        print("❌ Skipping - Raindrop or Gemini not available")
        return
    
    model = genai.GenerativeModel("gemini-2.5-pro")
    convo_id = f"convo_{int(time.time())}"
    user_id = "test_user_001"
    
    # Conversation turns
    turns = [
        "I'm building a support chatbot. What features should it have?",
        "How should I handle user frustration?",
        "Can you give me an example response for an angry user?"
    ]
    
    history = []
    
    for i, user_msg in enumerate(turns):
        print(f"\n--- Turn {i+1} ---")
        print(f"User: {user_msg}")
        
        # Build conversation context
        history.append({"role": "user", "parts": [user_msg]})
        
        # Get AI response
        chat = model.start_chat(history=history[:-1] if len(history) > 1 else [])
        response = chat.send_message(user_msg)
        ai_output = response.text
        
        history.append({"role": "model", "parts": [ai_output]})
        
        print(f"AI: {ai_output[:200]}...")
        
        # Track in Raindrop
        event_id = f"evt_{convo_id}_{i}"
        raindrop.track_ai(
            user_id=user_id,
            event="chatbot_turn",
            event_id=event_id,
            model="gemini-2.5-pro",
            input=user_msg,
            output=ai_output,
            convo_id=convo_id,
            properties={
                "turn_number": i + 1,
                "total_tokens": len(user_msg.split()) + len(ai_output.split())
            }
        )
        
        # Simulate user giving feedback on last turn
        if i == len(turns) - 1:
            raindrop.track_signal(
                event_id=event_id,
                name="thumbs_up",
                signal_type="default",
                sentiment="POSITIVE"
            )
            print("✅ User gave thumbs up!")
    
    raindrop.flush()
    print(f"\n✅ Completed {len(turns)}-turn conversation")


def test_error_scenario():
    """Test tracking an error/failure scenario"""
    print("\n" + "="*60)
    print("TEST 5: Error/Failure Scenario")
    print("="*60)
    
    if not RAINDROP_AVAILABLE:
        print("❌ Skipping - Raindrop not available")
        return
    
    event_id = f"evt_error_{int(time.time())}"
    
    # Track a failed interaction
    raindrop.track_ai(
        user_id="test_user_001",
        event="tool_call_failed",
        event_id=event_id,
        model="gemini-2.5-pro",
        input="Book me a flight to Tokyo",
        output="I'm sorry, I encountered an error while trying to search for flights. The booking API is currently unavailable.",
        properties={
            "error": True,
            "error_type": "api_timeout",
            "tool_name": "flight_search"
        }
    )
    print("✅ Tracked failed interaction")
    
    # Track negative signal
    raindrop.track_signal(
        event_id=event_id,
        name="thumbs_down",
        signal_type="default",
        sentiment="NEGATIVE"
    )
    print("✅ Tracked thumbs_down signal")
    
    # Track user frustration feedback
    raindrop.track_signal(
        event_id=event_id,
        name="user_frustration",
        signal_type="feedback",
        comment="This is the third time the booking hasn't worked!"
    )
    print("✅ Tracked user frustration")
    
    raindrop.flush()


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🌧️  RAINDROP SDK TEST")
    print("="*60)
    
    if not RAINDROP_AVAILABLE:
        print("\n❌ Cannot run tests - Raindrop SDK not installed")
        print("   Run: pip install raindrop-ai")
        return
    
    write_key = os.getenv("RAINDROP_WRITE_KEY")
    if not write_key:
        print("\n❌ Cannot run tests - RAINDROP_WRITE_KEY not set")
        print("   Get your key at: https://app.raindrop.ai")
        return
    
    print(f"\n✅ Raindrop SDK available")
    print(f"✅ Write key configured: {write_key[:10]}...")
    print(f"{'✅' if GEMINI_AVAILABLE else '❌'} Gemini API {'available' if GEMINI_AVAILABLE else 'not available'}")
    
    try:
        test_basic_tracking()
        test_signals()
        test_partial_events()
        test_complex_conversation()
        test_error_scenario()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED!")
        print("="*60)
        print("\nCheck your Raindrop dashboard at https://app.raindrop.ai")
        print("to see the tracked events and signals.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise
    finally:
        if RAINDROP_AVAILABLE:
            raindrop.shutdown()
            print("\n✅ Raindrop shutdown complete")


if __name__ == "__main__":
    main()

