import streamlit as st
import os
from dotenv import load_dotenv
from typing import TypedDict, List, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
import time

# Load environment variables
load_dotenv()

# Define the state structure
class GraphState(TypedDict):
    topic: str
    search_results: List[dict]
    summary: str
    decision: Literal["Sufficient", "Insufficient"]
    iteration: int
    log_messages: List[str]

# Initialize components
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
    state["log_messages"].append(f"🔍 **Search Node (Iteration {state['iteration']}):** Found {len(search_results)} sources")
    
    return state

def summarize_node(state: GraphState) -> GraphState:
    """Generate a summary from search results"""
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash-exp"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are a professional research analyst. Create a comprehensive, high-quality summary about: {topic}
    
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
        f"Title: {result.get('title', 'No title')}\nContent: {result.get('content', 'No content')}\n"
        for result in state["search_results"]
    ])
    
    summary = chain.invoke({
        "topic": state["topic"],
        "search_results": search_content
    })
    
    state["summary"] = summary
    state["log_messages"].append(f"📝 **Summarize Node:** Generated summary ({len(summary)} characters)")
    
    return state

def evaluate_node(state: GraphState) -> GraphState:
    """Evaluate if the summary is good enough"""
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash-exp"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1  # Lower temperature for more consistent evaluation
    )
    
    # Force at least 2 iterations to demonstrate self-correction
    if state["iteration"] < 3:
        # Be more strict for first 2 iterations to force improvement
        prompt = ChatPromptTemplate.from_template("""
        You are a very strict quality evaluator for research summaries. Evaluate this summary about "{topic}":
        
        SUMMARY TO EVALUATE:
        {summary}
        
        SEARCH RESULTS USED:
        {search_results}
        
        CURRENT ITERATION: {iteration}
        
        Evaluation Criteria (be VERY strict for iterations 1-2):
        1. Is the summary comprehensive and detailed (at least 300 words)?
        2. Does it include multiple specific insights, data, statistics, or examples?
        3. Are the search results diverse and of high quality?
        4. Does the summary provide deep analysis, not just surface information?
        5. For iterations 1-2: Be extra critical and demand more depth
        
        Respond with ONLY one word:
        - "Sufficient" if the summary meets VERY high quality standards (rare for early iterations)
        - "Insufficient" if it needs improvement (be strict for iterations 1-2)
        
        Decision:""")
    else:
        # Be more lenient for iteration 3+
        prompt = ChatPromptTemplate.from_template("""
        You are a quality evaluator for research summaries. Evaluate this summary about "{topic}":
        
        SUMMARY TO EVALUATE:
        {summary}
        
        SEARCH RESULTS USED:
        {search_results}
        
        CURRENT ITERATION: {iteration}
        
        Evaluation Criteria (reasonable standards for iteration 3+):
        1. Is the summary comprehensive and detailed (at least 200 words)?
        2. Does it include specific insights, data, or examples?
        3. Are the search results relevant?
        4. Does the summary adequately cover the topic?
        
        Respond with ONLY one word:
        - "Sufficient" if the summary meets quality standards
        - "Insufficient" if it needs improvement
        
        Decision:""")
    
    chain = prompt | llm | StrOutputParser()
    
    search_content = "\n".join([
        f"Title: {result.get('title', 'No title')}\nContent: {result.get('content', 'No content')[:200]}..."
        for result in state["search_results"]
    ])
    
    decision = chain.invoke({
        "topic": state["topic"],
        "summary": state["summary"],
        "search_results": search_content,
        "iteration": state["iteration"]
    }).strip()
    
    # Force insufficient for first 2 iterations to demonstrate self-correction
    if state["iteration"] <= 2:
        decision = "Insufficient"
        state["log_messages"].append(f"🤔 **Evaluate Node:** Iteration {state['iteration']} - Being strict to demonstrate improvement")
    else:
        # Ensure we get a valid decision for iteration 3+
        if "Sufficient" in decision:
            decision = "Sufficient"
        else:
            decision = "Insufficient"
    
    state["decision"] = decision
    state["log_messages"].append(f"🤔 **Evaluate Node:** Quality assessment = {decision}")
    
    if decision == "Insufficient":
        state["iteration"] += 1
        state["log_messages"].append(f"🔄 **Decision:** Research needs improvement. Starting iteration {state['iteration']}...")
    else:
        state["log_messages"].append(f"✅ **Decision:** Research meets quality standards. Analysis complete!")
    
    return state

def should_continue(state: GraphState) -> str:
    """Decide whether to continue or end based on evaluation"""
    if state["decision"] == "Insufficient" and state["iteration"] <= 4:  # Max 4 iterations
        return "search"
    else:
        return END

def create_graph():
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

def main():
    st.set_page_config(
        page_title="LangGraph Self-Correcting Researcher",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 LangGraph: The Self-Correcting Researcher")
    st.markdown("*Intelligent research with self-evaluation and iteration*")
    
    # Educational info box
    st.info("""
    **What is this tool?** This demonstrates LangGraph's intelligent, cyclical approach. 
    It can **evaluate its own work** and **loop back** to improve results if needed.
    It acts like a real researcher who reviews and refines their work.
    """)
    
    # Input section
    st.subheader("🔬 Enter Your Research Topic")
    topic = st.text_area(
        "What would you like comprehensive research about?",
        placeholder="e.g., Market viability for quantum computing in logistics, Impact of AI on healthcare diagnostics, etc.",
        height=100
    )
    
    if st.button("🚀 Start Research", type="primary"):
        if not topic.strip():
            st.warning("Please enter a topic to research!")
            return
        
        # Initialize the graph
        try:
            graph = create_graph()
        except Exception as e:
            st.error(f"Failed to initialize research graph: {str(e)}")
            return
        
        # Create placeholders for real-time updates
        log_placeholder = st.empty()
        progress_placeholder = st.empty()
        
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
        try:
            with st.status("🧠 Research Agent Thinking...", expanded=True) as status:
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
                
                status.update(label="✅ Research Complete!", state="complete")
                
                # Get final state
                final_summary = current_state.get("summary", "No summary generated")
                final_results = current_state.get("search_results", [])
                total_iterations = current_state.get("iteration", 1)
                
        except Exception as e:
            st.error(f"An error occurred during research: {str(e)}")
            return
        
        # Display results
        st.markdown("---")
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
        
        # Show source information
        with st.expander("🔗 View Research Sources"):
            for i, result in enumerate(final_results, 1):
                st.write(f"**Source {i}:** {result.get('title', 'No title')}")
                st.write(f"**URL:** {result.get('url', 'No URL')}")
                st.write(f"**Content:** {result.get('content', 'No content')[:300]}...")
                st.write("---")
    
    # Educational explanation
    st.markdown("---")
    st.markdown("### 🎯 What Just Happened?")
    st.markdown("""
    This tool demonstrated **LangGraph's intelligent, cyclical workflow**:
    
    1. **🔍 Search:** Found initial sources about your topic
    2. **📝 Summarize:** Generated a comprehensive summary
    3. **🤔 Evaluate:** AI evaluated its own work quality
    4. **🔄 Loop Back:** If quality was insufficient, it searched again with refined queries
    5. **✅ Complete:** Finished when quality standards were met
    
    **Key Characteristics of LangGraph:**
    - ✅ **Self-correcting** - Can evaluate and improve its own output
    - ✅ **Adaptive** - Changes search strategy based on previous results
    - ✅ **Quality-focused** - Doesn't stop until standards are met
    - ✅ **Intelligent loops** - Makes decisions about when to continue or stop
    - ❌ **Slower** - Takes more time due to evaluation and potential iterations
    - ❌ **More complex** - Requires more sophisticated logic
    
    This makes LangGraph perfect for **high-quality, comprehensive research** where accuracy and depth matter more than speed.
    """)

if __name__ == "__main__":
    main()