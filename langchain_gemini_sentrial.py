"""
Complex LangChain agent with 10+ steps for testing Sentrial tracking.

This simulates a realistic customer support agent that needs to:
1. Look up user info
2. Check order history
3. Search knowledge base
4. Check refund policies
5. Check inventory
6. Create support ticket
7. Check warranty status
8. Calculate refund amount
9. Process refund
10. Send confirmation email

Install dependencies:
    pip install langchain langchain-google-genai python-dotenv

Set your keys in .env:
    GOOGLE_API_KEY=your_gemini_key_here
    SENTRIAL_API_KEY=your_sentrial_api_key_here
"""

import sys
import os
import random
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add python-sdk to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../packages/python-sdk'))

from sentrial import SentrialClient, SentrialCallbackHandler
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI


# =============================================================================
# TOOLS - Simulating a realistic support system with latency
# =============================================================================

@tool
def get_customer_info(customer_id: str) -> str:
    """Get detailed customer information including account status, tier, and history.
    Input should be a customer ID like 'CUST-12345'."""
    time.sleep(0.4)  # Simulate DB lookup
    return f"""Customer {customer_id}:
    - Name: John Smith
    - Email: john.smith@email.com
    - Account Tier: Premium Gold
    - Member Since: January 2021
    - Total Orders: 47
    - Lifetime Value: $3,847.50
    - Support Priority: High
    - Notes: VIP customer, always resolve issues quickly"""


@tool
def get_order_history(customer_id: str, limit: int = 5) -> str:
    """Get recent order history for a customer.
    Input should be customer ID. Returns most recent orders."""
    time.sleep(0.5)  # Simulate DB query
    orders = [
        {"id": "ORD-98765", "date": "2024-01-15", "amount": "$299.99", "status": "Delivered", "item": "Wireless Headphones Pro"},
        {"id": "ORD-98432", "date": "2024-01-10", "amount": "$149.50", "status": "Delivered", "item": "Smart Watch Band"},
        {"id": "ORD-98101", "date": "2024-01-05", "amount": "$599.00", "status": "Delivered", "item": "4K Monitor 27inch"},
        {"id": "ORD-97888", "date": "2023-12-20", "amount": "$89.99", "status": "Delivered", "item": "USB-C Hub"},
        {"id": "ORD-97654", "date": "2023-12-15", "amount": "$1,299.00", "status": "Delivered", "item": "Laptop Stand Deluxe"},
    ]
    result = f"Recent orders for {customer_id}:\n"
    for order in orders[:limit]:
        result += f"  - {order['id']}: {order['item']} - {order['amount']} ({order['status']}) - {order['date']}\n"
    return result


@tool
def get_order_details(order_id: str) -> str:
    """Get detailed information about a specific order.
    Input should be an order ID like 'ORD-98765'."""
    time.sleep(0.3)  # Simulate lookup
    return f"""Order {order_id} Details:
    - Product: Wireless Headphones Pro (SKU: WHP-2024-BLK)
    - Price: $299.99
    - Tax: $24.00
    - Shipping: Free (Premium member)
    - Total: $323.99
    - Order Date: January 15, 2024
    - Delivery Date: January 18, 2024
    - Shipping Address: 123 Main St, San Francisco, CA 94102
    - Payment: Visa ending in 4242
    - Warranty: 2-year manufacturer warranty (expires Jan 2026)
    - Return Window: 30 days (expires Feb 14, 2024)"""


@tool
def search_knowledge_base(query: str) -> str:
    """Search the support knowledge base for relevant articles.
    Input should be a search query about the issue."""
    time.sleep(0.6)  # Simulate search
    articles = {
        "refund": [
            "KB-1001: How to Request a Refund",
            "KB-1002: Refund Processing Times",
            "KB-1003: Refund Policy for Premium Members",
        ],
        "headphones": [
            "KB-2001: Troubleshooting Wireless Headphones",
            "KB-2002: Pairing Instructions for Headphones Pro",
            "KB-2003: Common Audio Quality Issues",
        ],
        "warranty": [
            "KB-3001: Understanding Your Warranty",
            "KB-3002: Filing a Warranty Claim",
            "KB-3003: Extended Warranty Options",
        ],
        "default": [
            "KB-0001: Contact Support Options",
            "KB-0002: Frequently Asked Questions",
            "KB-0003: Shipping and Returns Overview",
        ]
    }
    
    for key, arts in articles.items():
        if key in query.lower():
            return f"Found relevant articles for '{query}':\n" + "\n".join(f"  - {a}" for a in arts)
    
    return f"Found articles for '{query}':\n" + "\n".join(f"  - {a}" for a in articles["default"])


@tool
def check_refund_policy(order_id: str, reason: str) -> str:
    """Check if an order is eligible for refund and what policy applies.
    Inputs: order_id and reason for refund."""
    time.sleep(0.4)  # Simulate policy check
    return f"""Refund Policy Check for {order_id}:
    ✓ Within 30-day return window: YES
    ✓ Product category eligible: YES (Electronics)
    ✓ Customer tier allows full refund: YES (Premium Gold)
    
    Applicable Policy: FULL_REFUND_PREMIUM
    - Full purchase price refundable
    - Original shipping: Refundable (Premium benefit)
    - Return shipping: Free prepaid label provided
    - Processing time: 3-5 business days
    - Refund method: Original payment method
    
    Reason category: {reason}
    Auto-approval: ELIGIBLE (Premium tier + valid window)"""


@tool
def check_inventory(sku: str) -> str:
    """Check current inventory and availability for a product.
    Input should be a product SKU."""
    time.sleep(0.3)  # Simulate inventory check
    return f"""Inventory Status for {sku}:
    - In Stock: Yes
    - Warehouse Qty: 1,247 units
    - Fulfillment Centers:
      * West Coast (CA): 523 units
      * East Coast (NJ): 412 units  
      * Central (TX): 312 units
    - Restock ETA: N/A (sufficient stock)
    - Replacement Available: Yes
    - Exchange Processing: Same-day if before 2pm PT"""


@tool
def check_warranty_status(order_id: str) -> str:
    """Check warranty status and coverage for an order.
    Input should be an order ID."""
    time.sleep(0.35)  # Simulate warranty lookup
    return f"""Warranty Status for {order_id}:
    - Warranty Type: Manufacturer Standard
    - Coverage Period: 2 years
    - Start Date: January 18, 2024
    - Expiration: January 18, 2026
    - Status: ACTIVE
    - Claims Filed: 0
    - Coverage Includes:
      * Manufacturing defects
      * Component failures
      * Battery issues (1 year)
    - NOT Covered:
      * Physical damage
      * Water damage
      * Lost/stolen items
    - Extended Warranty: Available for $49.99 (adds 1 year)"""


@tool
def calculate_refund(order_id: str, include_shipping: bool = True) -> str:
    """Calculate the refund amount for an order.
    Inputs: order_id and whether to include shipping."""
    time.sleep(0.25)  # Simulate calculation
    return f"""Refund Calculation for {order_id}:
    - Product Price: $299.99
    - Original Tax: $24.00
    - Shipping Paid: $0.00 (Free - Premium)
    
    Subtotal: $323.99
    
    Deductions:
    - Restocking Fee: $0.00 (Premium member waived)
    - Return Shipping: $0.00 (Prepaid label)
    
    TOTAL REFUND: $323.99
    
    Refund Breakdown:
    - To Visa ****4242: $323.99
    - Store Credit: $0.00
    
    Processing: 3-5 business days after item received"""


@tool
def create_support_ticket(
    customer_id: str,
    category: str,
    priority: str,
    summary: str
) -> str:
    """Create a support ticket in the system.
    Inputs: customer_id, category, priority (low/medium/high/urgent), summary."""
    time.sleep(0.45)  # Simulate ticket creation
    ticket_id = f"TKT-{random.randint(100000, 999999)}"
    return f"""Support Ticket Created:
    - Ticket ID: {ticket_id}
    - Customer: {customer_id}
    - Category: {category}
    - Priority: {priority.upper()}
    - Status: OPEN
    - Created: {time.strftime('%Y-%m-%d %H:%M:%S')}
    - Summary: {summary}
    - Assigned To: Auto-routing based on category
    - SLA: {'4 hours' if priority == 'urgent' else '24 hours' if priority == 'high' else '48 hours'}
    - Customer Notified: Yes (email sent)"""


@tool
def process_refund(order_id: str, amount: float, reason: str) -> str:
    """Process a refund for an order. This initiates the actual refund.
    Inputs: order_id, amount, and reason."""
    time.sleep(0.8)  # Simulate payment processing
    refund_id = f"REF-{random.randint(100000, 999999)}"
    return f"""Refund Processed Successfully:
    - Refund ID: {refund_id}
    - Order: {order_id}
    - Amount: ${amount:.2f}
    - Reason: {reason}
    - Status: INITIATED
    - Payment Method: Visa ****4242
    - Expected Completion: 3-5 business days
    - Confirmation Email: Sent to customer
    - Return Label: Generated and emailed
    
    Next Steps:
    1. Customer receives return shipping label
    2. Customer ships item within 14 days
    3. Warehouse receives and inspects item
    4. Refund completes to original payment method"""


@tool
def send_customer_email(
    customer_id: str,
    template: str,
    custom_message: str = ""
) -> str:
    """Send an email to the customer using a template.
    Inputs: customer_id, template name, optional custom message."""
    time.sleep(0.3)  # Simulate email sending
    templates = {
        "refund_confirmation": "Refund Request Confirmed",
        "return_label": "Your Return Shipping Label",
        "ticket_created": "Support Ticket Created",
        "resolution_summary": "Your Issue Has Been Resolved",
    }
    subject = templates.get(template, "Update from Support")
    return f"""Email Sent Successfully:
    - To: john.smith@email.com ({customer_id})
    - Subject: {subject}
    - Template: {template}
    - Custom Message: {custom_message if custom_message else 'N/A'}
    - Sent At: {time.strftime('%Y-%m-%d %H:%M:%S')}
    - Delivery Status: Delivered
    - Tracking ID: EMAIL-{random.randint(10000, 99999)}"""


@tool
def escalate_to_specialist(
    ticket_id: str,
    department: str,
    reason: str
) -> str:
    """Escalate a ticket to a specialist department.
    Inputs: ticket_id, department (billing/technical/management), reason."""
    time.sleep(0.35)  # Simulate escalation
    return f"""Ticket Escalated:
    - Ticket: {ticket_id}
    - Escalated To: {department.title()} Team
    - Reason: {reason}
    - New Priority: URGENT
    - New SLA: 2 hours
    - Specialist Assigned: Yes (next available)
    - Customer Notified: Yes
    - Original Agent: Retained as secondary"""


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a customer support representative handling service requests.
Use a direct, matter-of-fact tone focused on problem resolution.
Provide factual information without emotional commentary.
State policies and procedures clearly without embellishment.
Keep responses brief and solution-oriented."""

# =============================================================================
# MAIN AGENT
# =============================================================================

def main():
    print("=" * 70)
    print("🤖 Complex Support Agent - Testing Sentrial with 10+ Tool Calls")
    print("=" * 70)
    print()
    
    # Check for API keys
    if not os.environ.get("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)
    
    api_key = os.environ.get("SENTRIAL_API_KEY")
    if not api_key:
        print("⚠️  Warning: SENTRIAL_API_KEY not set")
    
    # Initialize Sentrial client
    client = SentrialClient(
        api_url=os.environ.get("SENTRIAL_API_URL", "http://localhost:3001"),
        api_key=api_key
    )
    
    AGENT_NAME = "gemini_support_agent"
    
    # Create session
    print("📝 Creating Sentrial session...")
    session_id = client.create_session(
        name="Complex Multi-Step Support Agent Test",
        agent_name=AGENT_NAME,
    )
    print(f"✓ Session ID: {session_id}\n")
    
    # Create callback handler
    sentrial_handler = SentrialCallbackHandler(
        client=client,
        session_id=session_id,
        verbose=True
    )
    
    # All our tools
    tools = [
        get_customer_info,
        get_order_history,
        get_order_details,
        search_knowledge_base,
        check_refund_policy,
        check_inventory,
        check_warranty_status,
        calculate_refund,
        create_support_ticket,
        process_refund,
        send_customer_email,
        escalate_to_specialist,
    ]
    
    # Initialize Gemini
    print("🔧 Initializing Gemini 3 Pro...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-pro",
        temperature=0,
    )
    
    # Create agent
    agent = create_agent(llm, tools)
    
    # Complex multi-step query that requires many tool calls
    print("\n" + "=" * 70)
    print("📨 Customer Request:")
    print("=" * 70)
    
    customer_request = """
    I'm customer CUST-12345 and I need help with a refund for my recent order.
    
    I bought the Wireless Headphones Pro about 2 weeks ago (order ORD-98765) 
    and they stopped working - no sound comes out of the left ear.
    
    Can you:
    1. Look up my account and verify I'm a premium member
    2. Find my order details
    3. Check if it's still under warranty
    4. See what your refund policy says for this situation
    5. Check if you have replacement units in stock
    6. If eligible, process a full refund for me
    7. Create a support ticket to document this
    8. Send me a confirmation email with the return label
    
    I'd really appreciate a quick resolution as I need headphones for work.
    """
    
    print(customer_request)
    print("=" * 70)
    print()
    
    print("🚀 Running agent... (this will make 10+ tool calls)\n")
    start_time = time.time()
    
    try:
        # Run the agent
        result = agent.invoke(
            {"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": customer_request}
            ]},
            config={"callbacks": [sentrial_handler]}
        )
        
        elapsed = time.time() - start_time
        usage = sentrial_handler.get_usage_summary()
        
        print("\n" + "=" * 70)
        print("📊 AGENT RESPONSE:")
        print("=" * 70)
        
        if "messages" in result:
            final_msg = result["messages"][-1]
            print(final_msg.content)
        else:
            print(result)
        
        print("\n" + "=" * 70)
        print("📈 SENTRIAL CAPTURED METRICS:")
        print("=" * 70)
        print(f"  ⏱️  Total Time: {elapsed:.2f}s")
        print(f"  🔄 LLM Calls: {usage['llm_calls']}")
        print(f"  📝 Prompt Tokens: {usage['total_prompt_tokens']:,}")
        print(f"  💬 Completion Tokens: {usage['total_completion_tokens']:,}")
        print(f"  📊 Total Tokens: {usage['total_tokens']:,}")
        print(f"  💰 Estimated Cost: ${usage['total_cost']:.4f}")
        print(f"  ⏳ Duration: {usage['duration_ms']}ms")
        print("=" * 70)
        
        # Complete session with rich metrics
        client.complete_session(
            session_id=session_id,
            success=True,
            duration_ms=usage['duration_ms'],
            estimated_cost=usage['total_cost'],
            prompt_tokens=usage['total_prompt_tokens'],
            completion_tokens=usage['total_completion_tokens'],
            total_tokens=usage['total_tokens'],
            custom_metrics={
                "customer_satisfaction_score": 4.5,
                "first_contact_resolution": 1,
                "escalation_rate": 0,
                "average_handle_time": int(elapsed),
                "tools_called": usage['llm_calls'],
            }
        )
        
        print("\n✅ Session completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        usage = sentrial_handler.get_usage_summary()
        client.complete_session(
            session_id=session_id,
            success=False,
            failure_reason=str(e),
            duration_ms=usage['duration_ms'],
            estimated_cost=usage['total_cost'],
            prompt_tokens=usage['total_prompt_tokens'],
            completion_tokens=usage['total_completion_tokens'],
            total_tokens=usage['total_tokens'],
        )
    
    print(f"\n🎉 View your session at:")
    print(f"   http://localhost:3000/sentrial-yc-w26/agents/{AGENT_NAME}")


if __name__ == "__main__":
    main()

