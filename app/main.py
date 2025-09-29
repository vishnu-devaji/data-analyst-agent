import os
import json
import base64
import io
import re
from typing import List, Optional, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import openai
from openai import OpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from urllib.parse import urljoin, urlparse
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Data Analyst Agent API",
    description="AI-powered data analysis, visualization, and web scraping API",
    version="1.0.0"
)

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    logger.info("Please ensure OPENAI_API_KEY environment variable is set")
    client = None

class DataAnalystAgent:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    async def scrape_url(self, url: str) -> pd.DataFrame:
        """Scrape data from URL and return as DataFrame"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to find tables first
            tables = pd.read_html(response.content)
            if tables:
                # Return the largest table (most likely to contain the main data)
                largest_table = max(tables, key=len)
                logger.info(f"Scraped table with {len(largest_table)} rows from {url}")
                return largest_table
            
            # If no tables, extract text content
            text_content = soup.get_text(strip=True)
            logger.info(f"Scraped text content from {url}")
            return pd.DataFrame({'content': [text_content]})
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to scrape URL: {str(e)}")
    
    def create_visualization(self, data: pd.DataFrame, plot_type: str, 
                           x_col: str = None, y_col: str = None, 
                           title: str = "Data Visualization") -> str:
        """Create visualization and return as base64 encoded data URI"""
        try:
            plt.figure(figsize=(10, 6))
            plt.style.use('default')
            
            if plot_type.lower() == 'scatter' or 'scatterplot' in plot_type.lower():
                if x_col and y_col and str(x_col) in [str(c) for c in data.columns] and str(y_col) in [str(c) for c in data.columns]:
                    # Clean data - convert to numeric if possible
                    x_data = pd.to_numeric(data[x_col], errors='coerce').dropna()
                    y_data = pd.to_numeric(data[y_col], errors='coerce').dropna()
                    
                    # Align the data
                    min_len = min(len(x_data), len(y_data))
                    x_data = x_data.iloc[:min_len]
                    y_data = y_data.iloc[:min_len]
                    
                    plt.scatter(x_data, y_data, alpha=0.6)
                    
                    # Add regression line if requested
                    if 'regression' in plot_type.lower() or 'trend' in plot_type.lower():
                        z = np.polyfit(x_data, y_data, 1)
                        p = np.poly1d(z)
                        plt.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
                    
                    plt.xlabel(str(x_col))
                    plt.ylabel(str(y_col))
                else:
                    # Default scatter with first two numeric columns
                    numeric_cols = data.select_dtypes(include=[np.number]).columns[:2]
                    if len(numeric_cols) >= 2:
                        plt.scatter(data[numeric_cols[0]], data[numeric_cols[1]], alpha=0.6)
                        plt.xlabel(str(numeric_cols[0]))
                        plt.ylabel(str(numeric_cols[1]))
            
            elif plot_type.lower() in ['bar', 'histogram']:
                if x_col and str(x_col) in [str(c) for c in data.columns]:
                    data[x_col].value_counts().head(20).plot(kind='bar')
                else:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        numeric_col = numeric_cols[0]
                        plt.hist(data[numeric_col].dropna(), bins=20, alpha=0.7)
            
            plt.title(title)
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            img_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            # Check size limit (100KB)
            if len(img_data) > 100000:
                # Reduce DPI and try again
                plt.figure(figsize=(8, 5))
                # Recreate plot with lower quality...
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=60, bbox_inches='tight')
                buffer.seek(0)
                img_data = buffer.getvalue()
                buffer.close()
                plt.close()
            
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return f"Error creating visualization: {str(e)}"
    
    def truncate_data_context(self, data_context: str, max_chars: int = 8000) -> str:
        """Truncate data context to prevent token overflow"""
        if len(data_context) <= max_chars:
            return data_context
        
        # Keep the beginning and add truncation notice
        truncated = data_context[:max_chars]
        truncated += f"\n\n[Data truncated - showing first {max_chars} characters of {len(data_context)} total]"
        return truncated
    
    async def analyze_with_openai(self, questions: str, data_context: str = None) -> List[str]:
        """Use OpenAI to analyze data and answer questions"""
        try:
            if client is None:
                return ["Error: OpenAI client not initialized. Please check your API key."]
            
            # Log what we're sending to OpenAI
            logger.info(f"Questions: {questions[:200]}...")
            logger.info(f"Data context length: {len(data_context) if data_context else 0}")
                
            system_prompt = """You are a data analyst AI. Answer questions about data directly and precisely.

Return ONLY a JSON array where each element answers the corresponding question:
- For numbers/counts: return as integer (e.g., 3)  
- For names/text: return as string (e.g., "Titanic")
- For correlations: return as decimal (e.g., -0.85)
- For visualizations: return placeholder "VISUALIZATION_NEEDED"

Example: [2, "Avatar", 0.92, "VISUALIZATION_NEEDED"]

No explanations, no code blocks, just the JSON array."""

            # Truncate data context to prevent token overflow
            truncated_context = self.truncate_data_context(data_context) if data_context else 'No data provided'

            user_prompt = f"""Data: {truncated_context}

Questions: {questions}

Answer as JSON array:"""

            logger.info("Sending request to OpenAI...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            logger.info(f"OpenAI raw response: {content}")
            
            # Clean up the response
            if content.startswith('```json'):
                content = content[7:-3].strip()
            elif content.startswith('```'):
                content = content[3:-3].strip()
            
            # Remove any leading/trailing whitespace and newlines
            content = content.strip()
            
            logger.info(f"Cleaned response: {content}")
            
            # Try to parse as JSON array
            try:
                answers = json.loads(content)
                if isinstance(answers, list):
                    logger.info(f"Successfully parsed {len(answers)} answers")
                    return answers
                else:
                    logger.warning(f"Response is not a list: {type(answers)}")
                    return [str(answers)]
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                logger.error(f"Content that failed to parse: '{content}'")
                
                # Try to extract array manually
                import re
                array_match = re.search(r'\[([^\[\]]*)\]', content)
                if array_match:
                    try:
                        parsed = json.loads('[' + array_match.group(1) + ']')
                        return parsed
                    except:
                        pass
                
                # Fallback: return as single item
                return [content] if content else ["No response from OpenAI"]
                
        except Exception as e:
            logger.error(f"Error with OpenAI analysis: {str(e)}")
            return [f"Error analyzing with OpenAI: {str(e)}"]
    
    async def process_request(self, questions_content: str, additional_files: List[UploadFile] = None) -> List[str]:
        """Main processing pipeline"""
        try:
            logger.info("Starting request processing...")
            
            # Parse questions
            questions = questions_content.strip()
            logger.info(f"Questions received: {questions[:200]}...")
            
            # Check if questions contain URLs to scrape
            urls = re.findall(r'https?://[^\s\n]+', questions)
            scraped_data = None
            data_context = ""
            
            if urls:
                logger.info(f"Found URLs to scrape: {urls}")
                for url in urls:
                    try:
                        logger.info(f"Scraping URL: {url}")
                        df = await self.scrape_url(url)
                        logger.info(f"Scraped data shape: {df.shape}")
                        
                        if scraped_data is None:
                            scraped_data = df
                        else:
                            scraped_data = pd.concat([scraped_data, df], ignore_index=True)
                        
                        # Create data context for OpenAI (truncate if too large)
                        data_context += f"\nData from {url}:\n"
                        data_context += f"Shape: {df.shape}\n"
                        data_context += f"Columns: {list(df.columns)}\n"
                        
                        # Limit sample data to prevent token overflow
                        sample_data = df.head(5).to_string()
                        if len(sample_data) > 1500:
                            sample_data = sample_data[:1500] + "...[truncated]"
                        data_context += f"Sample data:\n{sample_data}\n"
                        
                    except Exception as e:
                        logger.error(f"Failed to scrape {url}: {str(e)}")
                        continue
            else:
                logger.info("No URLs found in questions")
            
            # Process additional files if provided
            if additional_files:
                logger.info(f"Processing {len(additional_files)} additional files")
                for file in additional_files:
                    try:
                        if file.filename and file.filename.endswith('.csv'):
                            content = await file.read()
                            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                            if scraped_data is None:
                                scraped_data = df
                            else:
                                scraped_data = pd.concat([scraped_data, df], ignore_index=True)
                            data_context += f"\nData from {file.filename}:\n{df.head().to_string()}\n"
                    except Exception as e:
                        logger.error(f"Error processing file {file.filename}: {str(e)}")
                        continue
            
            logger.info(f"Total data context length: {len(data_context)}")
            
            # Use OpenAI to analyze and answer questions
            logger.info("Sending to OpenAI for analysis...")
            answers = await self.analyze_with_openai(questions, data_context)
            logger.info(f"Received {len(answers)} answers from OpenAI: {answers}")
            
            # Check if any answers require visualization
            final_answers = []
            for i, answer in enumerate(answers):
                # Ensure answer is a string
                answer_str = str(answer) if answer is not None else ""
                
                # Check if this answer is a visualization placeholder or if question asks for visualization
                if (answer_str == "VISUALIZATION_NEEDED" or 
                    any(vis_word in questions.lower() for vis_word in ['plot', 'chart', 'graph', 'visualiz', 'scatterplot'])):
                    
                    # Check if this specific question asks for visualization
                    question_lines = [q.strip() for q in questions.split('\n') if q.strip() and not q.strip().startswith('http') and '?' in q]
                    logger.info(f"Question lines: {question_lines}")
                    
                    if i < len(question_lines) and any(vis_word in str(question_lines[i]).lower() for vis_word in ['plot', 'chart', 'graph', 'visualiz', 'scatterplot']):
                        if scraped_data is not None and len(scraped_data) > 1:
                            logger.info(f"Creating visualization for question {i+1}")
                            
                            # Determine plot parameters from the question
                            plot_type = "scatter"
                            x_col = y_col = None
                            
                            # Try to identify columns for visualization
                            if 'rank' in str(question_lines[i]).lower() and 'peak' in str(question_lines[i]).lower():
                                # Look for Rank and Peak columns
                                rank_cols = [col for col in scraped_data.columns if 'rank' in str(col).lower()]
                                peak_cols = [col for col in scraped_data.columns if 'peak' in str(col).lower()]
                                logger.info(f"Found rank columns: {rank_cols}, peak columns: {peak_cols}")
                                if rank_cols and peak_cols:
                                    x_col, y_col = rank_cols[0], peak_cols[0]
                            
                            # Create visualization
                            viz_data_uri = self.create_visualization(
                                scraped_data, plot_type, x_col, y_col, 
                                title=f"Visualization for Question {i+1}"
                            )
                            final_answers.append(viz_data_uri)
                        else:
                            final_answers.append("Unable to create visualization: insufficient data")
                    else:
                        final_answers.append(answer_str)
                else:
                    final_answers.append(answer)
            
            logger.info(f"Final answers count: {len(final_answers)}")
            return final_answers
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            return [f"Error processing request: {str(e)}"]

# Initialize the agent
agent = DataAnalystAgent()

@app.post("/api/")
async def analyze_data(
    files: List[UploadFile] = File(...)
):
    """
    Main API endpoint for data analysis
    Accepts POST request with files including questions.txt and optional additional data files
    """
    try:
        questions_content = None
        additional_files = []
        
        # Process uploaded files
        for file in files:
            if file.filename == "questions.txt":
                content = await file.read()
                questions_content = content.decode('utf-8')
            else:
                additional_files.append(file)
        
        if not questions_content:
            raise HTTPException(status_code=400, detail="questions.txt file is required")
        
        # Process the request
        answers = await agent.process_request(questions_content, additional_files)
        
        return JSONResponse(content=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/debug")
async def debug_endpoint():
    """Debug endpoint to test OpenAI connection"""
    try:
        if client is None:
            return {"error": "OpenAI client not initialized", "api_key_set": bool(os.getenv("OPENAI_API_KEY"))}
        
        # Test simple OpenAI call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Return exactly this: [1, 2, 3]"}],
            temperature=0,
            max_tokens=50
        )
        
        content = response.choices[0].message.content.strip()
        
        return {
            "openai_connected": True,
            "test_response": content,
            "api_key_set": bool(os.getenv("OPENAI_API_KEY")),
            "client_initialized": client is not None
        }
    except Exception as e:
        return {"error": str(e), "api_key_set": bool(os.getenv("OPENAI_API_KEY"))}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Data Analyst Agent API is running"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Data Analyst Agent API",
        "description": "AI-powered data analysis, visualization, and web scraping",
        "endpoints": {
            "POST /api/": "Main analysis endpoint - upload questions.txt and optional data files",
            "GET /health": "Health check",
            "GET /": "This information page"
        },
        "usage": "Send POST request with questions.txt file containing analysis tasks"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)