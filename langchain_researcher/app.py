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

# Initialize tools and models
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

def create_chain(llm, search_tool):
    """Create a simple LangChain LCEL chain"""
    
    # Define the prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a professional research analyst. Based on the search results below, create a concise and informative summary about: {topic}
    
    Search Results:
    {search_results}
    
    Please provide a clear, well-structured summary that covers the key points from the search results.
    Focus on the most important and relevant information.
    """)
    
    def format_search_results(topic):
        """Format search results for the prompt"""
        search_response = search_tool.invoke(topic)
        results = search_response.get('results', [])
        formatted = "\n".join([
            f"Title: {result.get('title', 'No title')}\nContent: {result.get('content', 'No content')}\nURL: {result.get('url', 'No URL')}\n"
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

def main():
    st.set_page_config(
        page_title="LangChain Researcher",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 LangChain Researcher")
    st.markdown("*Fast, predictable research using a fixed workflow*")
    
    # Educational info box
    st.info("""
    **What is this tool?** This demonstrates LangChain's linear, assembly-line approach. 
    It follows a fixed 3-step recipe: **Search → Format → Research**. 
    It's fast and efficient but cannot adapt or retry if results are poor.
    """)
    
    # Initialize components
    try:
        llm, search_tool = initialize_components()
        chain = create_chain(llm, search_tool)
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.stop()
    
    # Input section
    st.subheader("📝 Enter Your Research Topic")
    topic = st.text_area(
        "What would you like a quick research about?",
        placeholder="e.g., Latest developments in AI, Market trends in renewable energy, etc.",
        height=100
    )
    
    if st.button("🔍 Generate Research", type="primary"):
        if not topic.strip():
            st.warning("Please enter a topic to research!")
            return
            
        # Show the fixed workflow steps
        with st.status("Following the recipe...", expanded=True) as status:
            st.write("**Step 1:** � SearRching the web for sources...")
            
            try:
                # Execute the search step
                search_response = search_tool.invoke(topic)
                search_results = search_response.get('results', [])
                st.write(f"✅ Found {len(search_results)} sources")
                
                st.write("**Step 2:** 📝 Formatting and analyzing results...")
                st.write("**Step 3:** 🤖 Generating research with Gemini...")
                
                # Execute the full chain
                summary = chain.invoke(topic)
                
                status.update(label="✅ Research Complete!", state="complete")
                
            except Exception as e:
                status.update(label="❌ Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")
                return
        
        # Display results
        st.subheader("📋 Research Results")
        
        # Show the summary
        st.markdown("### 📄 Generated Research")
        st.write(summary)
        
        # Show source information
        with st.expander("🔗 View Source Information"):
            for i, result in enumerate(search_results, 1):
                st.write(f"**Source {i}:** {result.get('title', 'No title')}")
                st.write(f"**URL:** {result.get('url', 'No URL')}")
                st.write(f"**Content:** {result.get('content', 'No content')[:200]}...")
                st.write("---")
    
    # Educational explanation
    st.markdown("---")
    st.markdown("### 🎯 What Just Happened?")
    st.markdown("""
    This tool followed a **fixed, predictable workflow**:
    
    1. **🔍 Search:** Used Tavily to find web sources about your topic
    2. **📝 Format:** Organized the search results into a structured format  
    3. **🤖 Summarize:** Used Gemini to generate a summary from the formatted data
    
    **Key Characteristics of LangChain:**
    - ✅ **Fast and efficient** - No decision-making overhead
    - ✅ **Predictable** - Always follows the same steps
    - ✅ **Simple to build** - Linear chain of operations
    - ❌ **Cannot adapt** - If search results are poor, it can't retry or change approach
    - ❌ **No self-correction** - Cannot evaluate its own output quality
    
    This makes LangChain perfect for **quick, "good enough" research** where speed matters more than perfection.
    """)

if __name__ == "__main__":
    main()