
# 🔬 LangChain vs LangGraph: Complete Developer Guide

A comprehensive, hands-on comparison between **LangChain** and **LangGraph** frameworks with detailed tutorials, code examples, and architectural insights for building AI applications.

## 📚 Table of Contents

1. [🎯 Project Overview](#-project-overview)
2. [🏗️ Architecture Comparison](#️-architecture-comparison)
3. [🚀 Quick Start Guide](#-quick-start-guide)
4. [📖 LangChain Deep Dive](#-langchain-deep-dive)
5. [🧠 LangGraph Deep Dive](#-langgraph-deep-dive)
6. [🛠️ Step-by-Step Tutorials](#️-step-by-step-tutorials)
7. [⚖️ Detailed Comparison](#️-detailed-comparison)
8. [🎯 When to Use What](#-when-to-use-what)
9. [🔧 Troubleshooting](#-troubleshooting)
10. [📈 Performance Analysis](#-performance-analysis)

---

## 🎯 Project Overview

This project demonstrates the fundamental differences between **LangChain** and **LangGraph** through interactive research applications that showcase their unique approaches to AI workflow orchestration.

### 🏗️ Project Structure

```
research_assistant_project/
├── langchain_researcher/     # Linear workflow demonstration
│   └── app.py               # LangChain researcher (assembly line)
├── langgraph_researcher/    # Cyclical workflow demonstration  
│   └── app.py              # LangGraph researcher (self-correcting)
├── requirements.txt        # Shared dependencies
├── .env                   # API configuration
├── run_apps.py           # Application launcher
└── README.md             # This comprehensive guide
```

### 🎭 What You'll Learn

- **Architectural Patterns**: Linear vs Cyclical workflow design
- **State Management**: Stateless vs Stateful application architecture
- **Decision Making**: Deterministic vs Adaptive execution models
- **Practical Implementation**: Real code examples and best practices
- **Performance Trade-offs**: Speed vs Quality considerations

---

## 🏗️ Architecture Comparison

### 🚀 LangChain: The Assembly Line Approach

**Core Philosophy**: Predictable, Linear Execution

```mermaid
graph LR
    A[Input] --> B[Search Web]
    B --> C[Format Results]
    C --> D[Generate Summary]
    D --> E[Output]
    
    style A fill:#e1f5fe
    style E fill:#e8f5e8
    style B,C,D fill:#fff3e0
```

**Key Characteristics:**
- **Fixed Pipeline**: Each step executes in predetermined order
- **No Backtracking**: Cannot revisit previous steps
- **Single Pass**: One execution from start to finish
- **Deterministic**: Same input always follows same path

### 🧠 LangGraph: The Intelligent Agent Approach

**Core Philosophy**: Adaptive, Self-Correcting Intelligence

```mermaid
graph TD
    A[Input] --> B[Search Web]
    B --> C[Generate Summary]
    C --> D[Evaluate Quality]
    D --> E{Good Enough?}
    E -->|No| F[Refine Search]
    F --> B
    E -->|Yes| G[Output]
    
    style A fill:#e1f5fe
    style G fill:#e8f5e8
    style B,C,F fill:#fff3e0
    style D fill:#f3e5f5
    style E fill:#ffebee
```

**Key Characteristics:**
- **Dynamic Routing**: Can loop back and improve
- **Self-Evaluation**: Assesses its own output quality
- **Iterative Improvement**: Gets better with each cycle
- **Adaptive Strategy**: Changes approach based on results

---

## 🚀 Quick Start Guide

### 1. **Environment Setup**

```bash
# Create virtual environment
python -m venv research_agent_env

# Activate environment
# Windows:
research_agent_env\Scripts\activate
# macOS/Linux:
source research_agent_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **API Configuration**

Create `.env` file in project root:

```env
# Required API Keys
TAVILY_API_KEY=your_tavily_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here  
GEMINI_CHAT_MODEL=gemini-2.0-flash-exp
```

**Getting API Keys:**
- **Tavily API**: Sign up at [tavily.com](https://tavily.com) for web search capabilities
- **Gemini API**: Get your key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### 3. **Run Applications**

```bash
# Interactive launcher
python run_apps.py

# Or run individually:
# LangChain: http://localhost:8501
cd langchain_researcher && streamlit run app.py --server.port 8501

# LangGraph: http://localhost:8502  
cd langgraph_researcher && streamlit run app.py --server.port 8502
```

---

## 📖 LangChain Deep Dive

### 🎯 What is LangChain?

LangChain is a framework for developing applications powered by language models. It follows a **linear, pipeline-based approach** where components are chained together in a fixed sequence.

### 🔧 Core Concepts

#### 1. **LCEL (LangChain Expression Language)**

The modern way to build LangChain applications using a declarative syntax:

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create a chain using LCEL
chain = (
    {
        "topic": RunnablePassthrough(),
        "search_results": format_search_results
    }
    | prompt_template
    | llm_model
    | output_parser
)

# Execute the chain
result = chain.invoke("AI in healthcare")
```

#### 2. **Components Breakdown**

```python
# 1. Prompt Template
prompt = ChatPromptTemplate.from_template("""
You are a professional research analyst. Based on the search results below, 
create a concise and informative summary about: {topic}

Search Results:
{search_results}

Please provide a clear, well-structured summary.
""")

# 2. LLM Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)

# 3. Output Parser
output_parser = StrOutputParser()

# 4. Custom Function
def format_search_results(topic):
    search_response = search_tool.invoke(topic)
    results = search_response.get('results', [])
    return "\n".join([
        f"Title: {result.get('title')}\nContent: {result.get('content')}\n"
        for result in results
    ])
```

### ✅ LangChain Strengths

- **🚀 Speed**: No decision-making overhead
- **📊 Predictability**: Always follows the same path
- **🔧 Simplicity**: Easy to build, debug, and maintain
- **💰 Cost-Effective**: Single-pass execution
- **🎯 Deterministic**: Same input = same process

### ❌ LangChain Limitations

- **🚫 No Quality Control**: Cannot evaluate its own output
- **🚫 No Adaptability**: Cannot change strategy if results are poor
- **🚫 No Self-Correction**: Cannot retry or refine approach
- **🚫 Fixed Logic**: Cannot handle edge cases dynamically

---

## 🧠 LangGraph Deep Dive

### 🎯 What is LangGraph?

LangGraph is a library for building stateful, multi-actor applications with LLMs. It enables **cyclical, graph-based workflows** where agents can make decisions, loop back, and self-correct.

### 🔧 Core Concepts

#### 1. **State Management**

LangGraph uses a shared state object that persists across all nodes:

```python
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    topic: str
    search_results: List[dict]
    summary: str
    decision: Literal["Sufficient", "Insufficient"]
    iteration: int
    log_messages: List[str]
```

#### 2. **Node Functions**

Each node is a function that takes state and returns modified state:

```python
def search_node(state: GraphState) -> GraphState:
    """Search for information about the topic"""
    search_tool = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"))
    
    # Modify search query based on iteration
    topic = state["topic"]
    if state["iteration"] > 1:
        topic = f"{topic} detailed analysis comprehensive research"
    
    search_response = search_tool.invoke(topic)
    search_results = search_response.get('results', [])
    
    # Update state
    state["search_results"] = search_results
    state["log_messages"].append(f"🔍 Found {len(search_results)} sources")
    
    return state

def summarize_node(state: GraphState) -> GraphState:
    """Generate a summary from search results"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3
    )
    
    prompt = ChatPromptTemplate.from_template("""
    Create a comprehensive, high-quality summary about: {topic}
    
    Search Results: {search_results}
    
    Requirements:
    - Provide detailed analysis with specific insights
    - Include key statistics and trends
    - Structure with clear sections
    - Aim for 200-300 words
    """)
    
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({
        "topic": state["topic"],
        "search_results": format_results(state["search_results"])
    })
    
    state["summary"] = summary
    return state

def evaluate_node(state: GraphState) -> GraphState:
    """Evaluate if the summary is good enough"""
    # AI-powered quality assessment
    if state["iteration"] < 3:
        decision = "Insufficient"  # Force improvement for demo
    else:
        decision = "Sufficient"
    
    state["decision"] = decision
    if decision == "Insufficient":
        state["iteration"] += 1
    
    return state
```

#### 3. **Conditional Routing**

LangGraph can make decisions about which node to execute next:

```python
def should_continue(state: GraphState) -> str:
    """Decide whether to continue or end based on evaluation"""
    if state["decision"] == "Insufficient" and state["iteration"] <= 4:
        return "search"  # Loop back to search
    else:
        return END  # Finish execution

# Add conditional edges
workflow.add_conditional_edges(
    "evaluate",
    should_continue,
    {
        "search": "search",  # Loop back if insufficient
        END: END            # Complete if sufficient
    }
)
```

#### 4. **Graph Construction**

```python
def create_langgraph_workflow():
    """Create the complete workflow"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("evaluate", evaluate_node)
    
    # Set entry point
    workflow.set_entry_point("search")
    
    # Add edges
    workflow.add_edge("search", "summarize")
    workflow.add_edge("summarize", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "search": "search",
            END: END
        }
    )
    
    return workflow.compile()
```

### ✅ LangGraph Strengths

- **🔄 Self-Correction**: Evaluates and improves its own output
- **🎯 Adaptive Strategy**: Changes approach based on results
- **✅ Quality Assurance**: Built-in quality control mechanisms
- **📈 Iterative Improvement**: Gets better with each cycle
- **🧠 Complex Reasoning**: Handles nuanced decision-making
- **🛡️ Robust Output**: Higher reliability and depth

### ❌ LangGraph Limitations

- **⏱️ Slower Execution**: Multiple iterations take time
- **🔧 Higher Complexity**: More sophisticated to build and debug
- **💰 Resource Intensive**: Multiple LLM calls per task
- **🔄 Potential Loops**: Risk of infinite iteration (requires safeguards)
- **💸 Cost**: More API calls = higher operational costs

---

## 🛠️ Step-by-Step Tutorials

### 🚀 Building a LangChain Application

#### Step 1: Setup Dependencies

```python
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()
```

#### Step 2: Initialize Components

```python
@st.cache_resource
def initialize_components():
    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash-exp"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3
    )
    
    # Initialize Tavily search tool
    search_tool = TavilySearch(
        api_key=os.getenv("TAVILY_API_KEY"),
        max_results=5
    )
    
    return llm, search_tool
```

#### Step 3: Create the Chain

```python
def create_chain(llm, search_tool):
    """Create a simple LangChain LCEL chain"""
    
    # Define the prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a professional research analyst. Based on the search results below, 
    create a concise and informative summary about: {topic}
    
    Search Results:
    {search_results}
    
    Please provide a clear, well-structured summary that covers the key points.
    """)
    
    def format_search_results(topic):
        """Format search results for the prompt"""
        search_response = search_tool.invoke(topic)
        results = search_response.get('results', [])
        formatted = "\n".join([
            f"Title: {result.get('title', 'No title')}\n"
            f"Content: {result.get('content', 'No content')}\n"
            f"URL: {result.get('url', 'No URL')}\n"
            for result in results
        ])
        return formatted
    
    # Create the chain using LCEL
    chain = (
        {
            "topic": RunnablePassthrough(),
            "search_results": format_search_results
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain
```

#### Step 4: Create Streamlit Interface

```python
def main():
    st.set_page_config(
        page_title="LangChain Researcher",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 LangChain Researcher")
    st.markdown("*Fast, predictable research using a fixed workflow*")
    
    # Initialize components
    llm, search_tool = initialize_components()
    chain = create_chain(llm, search_tool)
    
    # Input section
    topic = st.text_area(
        "What would you like a quick research about?",
        placeholder="e.g., Latest developments in AI",
        height=100
    )
    
    if st.button("🔍 Generate Research", type="primary"):
        if topic.strip():
            with st.status("Following the recipe...", expanded=True):
                st.write("**Step 1:** 🔍 Searching the web...")
                st.write("**Step 2:** 📝 Formatting results...")
                st.write("**Step 3:** 🤖 Generating summary...")
                
                # Execute the chain
                summary = chain.invoke(topic)
                
            st.subheader("📋 Research Results")
            st.write(summary)

if __name__ == "__main__":
    main()
```

### 🧠 Building a LangGraph Application

#### Step 1: Define State Structure

```python
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    topic: str
    search_results: List[dict]
    summary: str
    decision: Literal["Sufficient", "Insufficient"]
    iteration: int
    log_messages: List[str]
```

#### Step 2: Create Node Functions

```python
def search_node(state: GraphState) -> GraphState:
    """Search for information about the topic"""
    search_tool = TavilySearch(
        api_key=os.getenv("TAVILY_API_KEY"),
        max_results=5
    )
    
    # Modify search query based on iteration for better results
    topic = state["topic"]
    if state["iteration"] > 1:
        topic = f"{topic} detailed analysis comprehensive research"
    
    search_response = search_tool.invoke(topic)
    search_results = search_response.get('results', [])
    
    state["search_results"] = search_results
    state["log_messages"].append(
        f"🔍 Search (Iteration {state['iteration']}): Found {len(search_results)} sources"
    )
    
    return state

def summarize_node(state: GraphState) -> GraphState:
    """Generate a summary from search results"""
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash-exp"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are a professional research analyst. Create a comprehensive, 
    high-quality summary about: {topic}
    
    Search Results:
    {search_results}
    
    Requirements:
    - Provide detailed analysis with specific insights
    - Include key statistics, trends, or findings when available
    - Structure the summary with clear sections
    - Ensure the summary is thorough and informative
    - Aim for at least 200-300 words for comprehensive coverage
    """)
    
    chain = prompt | llm | StrOutputParser()
    
    search_content = "\n".join([
        f"Title: {result.get('title', 'No title')}\n"
        f"Content: {result.get('content', 'No content')}\n"
        for result in state["search_results"]
    ])
    
    summary = chain.invoke({
        "topic": state["topic"],
        "search_results": search_content
    })
    
    state["summary"] = summary
    state["log_messages"].append(
        f"📝 Summarize: Generated summary ({len(summary)} characters)"
    )
    
    return state

def evaluate_node(state: GraphState) -> GraphState:
    """Evaluate if the summary is good enough"""
    # Force at least 2 iterations to demonstrate self-correction
    if state["iteration"] < 3:
        decision = "Insufficient"
        state["log_messages"].append(
            f"🤔 Evaluate: Iteration {state['iteration']} - Needs improvement"
        )
    else:
        decision = "Sufficient"
        state["log_messages"].append("✅ Evaluate: Quality standards met")
    
    state["decision"] = decision
    
    if decision == "Insufficient":
        state["iteration"] += 1
        state["log_messages"].append(
            f"🔄 Starting iteration {state['iteration']}..."
        )
    else:
        state["log_messages"].append("✅ Research complete!")
    
    return state
```

#### Step 3: Define Routing Logic

```python
def should_continue(state: GraphState) -> str:
    """Decide whether to continue or end based on evaluation"""
    if state["decision"] == "Insufficient" and state["iteration"] <= 4:
        return "search"  # Loop back to search
    else:
        return END  # End the workflow
```

#### Step 4: Build the Graph

```python
def create_langgraph_workflow():
    """Create the LangGraph workflow"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("evaluate", evaluate_node)
    
    # Set entry point
    workflow.set_entry_point("search")
    
    # Add edges
    workflow.add_edge("search", "summarize")
    workflow.add_edge("summarize", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "search": "search",
            END: END
        }
    )
    
    return workflow.compile()
```

#### Step 5: Create Streamlit Interface with Streaming

```python
def main():
    st.set_page_config(
        page_title="LangGraph Self-Correcting Researcher",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 LangGraph: The Self-Correcting Researcher")
    st.markdown("*Intelligent research with self-evaluation and iteration*")
    
    # Input section
    topic = st.text_area(
        "What would you like comprehensive research about?",
        placeholder="e.g., Market viability for quantum computing in logistics",
        height=100
    )
    
    if st.button("🚀 Start Research", type="primary"):
        if topic.strip():
            # Initialize the graph
            graph = create_langgraph_workflow()
            
            # Create placeholders for real-time updates
            log_placeholder = st.empty()
            
            # Initialize state
            initial_state = {
                "topic": topic,
                "search_results": [],
                "summary": "",
                "decision": "Insufficient",
                "iteration": 1,
                "log_messages": []
            }
            
            # Execute the graph with streaming
            with st.status("🧠 Research Agent Thinking...", expanded=True):
                log_messages = []
                
                for event in graph.stream(initial_state):
                    # Get the current node and state
                    node_name = list(event.keys())[0]
                    current_state = event[node_name]
                    
                    # Update log messages
                    if "log_messages" in current_state:
                        log_messages = current_state["log_messages"]
                        
                        # Update the log display
                        log_text = "\n\n".join(log_messages)
                        log_placeholder.markdown(f"### 🔄 Agent Thought Process\n{log_text}")
                    
                    # Add a small delay to make the streaming visible
                    time.sleep(0.5)
                
                # Get final state
                final_summary = current_state.get("summary", "No summary generated")
                final_results = current_state.get("search_results", [])
                total_iterations = current_state.get("iteration", 1)
            
            # Display results
            st.subheader("📊 Research Results")
            
            # Show research statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔄 Iterations", total_iterations)
            with col2:
                st.metric("📚 Sources Found", len(final_results))
            with col3:
                st.metric("📝 Summary Length", f"{len(final_summary)} chars")
            
            # Show the final summary
            st.markdown("### 📄 Comprehensive Research Summary")
            st.write(final_summary)

if __name__ == "__main__":
    main()
```

---

## ⚖️ Detailed Comparison

### **State Management**
| Aspect | LangChain | LangGraph |
|--------|-----------|-----------|
| **State Handling** | Stateless pipeline | Persistent state object |
| **Data Flow** | Unidirectional | Bidirectional with loops |
| **Memory** | No memory between steps | Maintains context across iterations |
| **Debugging** | Linear trace | Complex state transitions |

### **Decision Making**
| Aspect | LangChain | LangGraph |
|--------|-----------|-----------|
| **Logic Type** | Predetermined | Dynamic/Conditional |
| **Branching** | None | Conditional edges |
| **Error Handling** | Fail fast | Retry and adapt |
| **Quality Control** | External only | Built-in evaluation |

### **Performance Characteristics**
| Metric | LangChain | LangGraph |
|--------|-----------|-----------|
| **Execution Time** | ~10-30 seconds | ~60-120 seconds |
| **API Calls** | 1 search + 1 LLM | 2-4 searches + 4-8 LLM |
| **Token Usage** | Low | High |
| **Reliability** | Consistent | Variable (but higher quality) |
| **Scalability** | High | Moderate |

### **Code Complexity**
| Aspect | LangChain | LangGraph |
|--------|-----------|-----------|
| **Lines of Code** | ~50-100 | ~200-300 |
| **Learning Curve** | Gentle | Steep |
| **Debugging Difficulty** | Easy | Moderate |
| **Maintenance** | Simple | Complex |

---

## 🎯 When to Use What

### **Choose LangChain When:**

✅ **Speed is more important than perfection**
- Customer support chatbots
- Quick content generation
- Real-time applications

✅ **You have consistent, predictable inputs**
- Batch processing tasks
- Template-based generation
- Standard workflows

✅ **Budget/resource constraints are tight**
- Startup projects
- High-volume, low-margin applications
- Cost-sensitive environments

✅ **You need to process large volumes quickly**
- Data processing pipelines
- Bulk content creation
- Mass personalization

✅ **The task is well-defined and straightforward**
- Simple Q&A systems
- Basic summarization
- Standard translations

✅ **You want simple, maintainable code**
- Small teams
- Rapid prototyping
- Educational projects

### **Choose LangGraph When:**

✅ **Quality and accuracy are paramount**
- Medical diagnosis assistance
- Legal document analysis
- Financial research

✅ **You're dealing with complex, nuanced tasks**
- Multi-step reasoning
- Research synthesis
- Strategic planning

✅ **You can afford higher computational costs**
- Enterprise applications
- High-value use cases
- Quality-critical systems

✅ **You need adaptive, intelligent behavior**
- Personal assistants
- Dynamic problem solving
- Context-aware applications

✅ **The task requires multiple perspectives or iterations**
- Academic research
- Comprehensive analysis
- Creative writing

✅ **You're building autonomous agents**
- Self-improving systems
- Adaptive workflows
- Intelligent automation

---

## 🔧 Troubleshooting

### **Common LangChain Issues**

#### Problem: Poor search results lead to bad output
```python
# Solution: Add input validation and search query optimization
def validate_and_optimize_query(topic):
    if len(topic.strip()) < 3:
        raise ValueError("Topic too short")
    
    # Add context for better search
    optimized = f"{topic} latest research findings analysis"
    return optimized

# Use in chain
chain = (
    {
        "topic": validate_and_optimize_query,
        "search_results": format_search_results
    }
    | prompt
    | llm
    | output_parser
)
```

#### Problem: Chain fails on certain inputs
```python
# Solution: Add error handling
def safe_chain_invoke(chain, topic):
    try:
        return chain.invoke(topic)
    except Exception as e:
        return f"Error processing request: {str(e)}"
```

### **Common LangGraph Issues**

#### Problem: Infinite loops in evaluation
```python
# Solution: Set maximum iteration limits
def should_continue(state: GraphState) -> str:
    max_iterations = 5  # Set hard limit
    if (state["decision"] == "Insufficient" and 
        state["iteration"] <= max_iterations):
        return "search"
    else:
        return END
```

#### Problem: State corruption between nodes
```python
# Solution: Use state validation
def validate_state(state: GraphState) -> GraphState:
    required_keys = ["topic", "search_results", "summary", "decision", "iteration"]
    for key in required_keys:
        if key not in state:
            state[key] = get_default_value(key)
    return state

# Apply in each node
def search_node(state: GraphState) -> GraphState:
    state = validate_state(state)
    # ... rest of function
```

### **General Issues**

#### Problem: API Rate Limits
```python
# Solution: Implement exponential backoff
import time
import random

def api_call_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "rate limit" in str(e).lower():
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries exceeded")
```

#### Problem: Token Limits
```python
# Solution: Chunk large inputs
def chunk_text(text, max_tokens=4000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
```

#### Problem: Cost Management
```python
# Solution: Monitor API usage
class CostTracker:
    def __init__(self):
        self.total_tokens = 0
        self.total_calls = 0
        self.cost_per_token = 0.0001  # Adjust based on model
    
    def track_call(self, tokens_used):
        self.total_tokens += tokens_used
        self.total_calls += 1
    
    def get_estimated_cost(self):
        return self.total_tokens * self.cost_per_token
    
    def should_continue(self, budget_limit):
        return self.get_estimated_cost() < budget_