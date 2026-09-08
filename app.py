import streamlit as st
import os
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import groq
from groq import Groq

# ============= Configuration =============
class StudyLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class StudyFormat(Enum):
    NOTES = "notes"
    FLASHCARDS = "flashcards"
    QUIZ = "quiz"
    SUMMARY = "summary"
    COMPREHENSIVE = "comprehensive"

@dataclass
class StudyPack:
    topic: str
    level: str
    format: str
    learning_objectives: List[str]
    content: Dict[str, str]
    assessment: Dict[str, any]
    review_feedback: Optional[str] = None
    final_version: Optional[str] = None
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

# ============= Workflow Stages =============
class WorkflowEngine:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.context = {}
        self.errors = []
        self.retry_count = 3
        
    def execute_stage(self, stage_name: str, stage_func, *args, **kwargs):
        """Execute a stage with error handling and retry logic"""
        for attempt in range(self.retry_count):
            try:
                st.write(f"🔄 Executing stage: {stage_name} (Attempt {attempt + 1})")
                result = stage_func(*args, **kwargs)
                if result:
                    return result
                else:
                    raise ValueError(f"Stage {stage_name} returned empty result")
            except Exception as e:
                error_msg = f"Error in {stage_name}: {str(e)}"
                self.errors.append({"stage": stage_name, "error": error_msg, "attempt": attempt + 1})
                st.warning(f"⚠️ {error_msg} - Retrying...")
                if attempt == self.retry_count - 1:
                    st.error(f"❌ Failed after {self.retry_count} attempts")
                    return None
                time.sleep(1)
        return None

    def planning_stage(self, topic: str, level: str, format: str) -> Dict:
        """Stage 1: Planning - Define learning objectives and structure"""
        prompt = f"""You are an expert educational planner. Create a detailed study plan for:
        Topic: {topic}
        Level: {level}
        Format: {format}

        Provide a structured response with:
        1. 5-8 clear learning objectives (numbered list)
        2. 3-5 main sections with subsections
        3. Estimated time allocation per section (in minutes)
        4. Prerequisites (if any)
        5. Key concepts to cover

        Format as JSON with keys: objectives, sections, time_allocation, prerequisites, key_concepts
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.3,
                max_tokens=2048
            )
            
            result_text = response.choices[0].message.content
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                planning_data = json.loads(json_match.group())
                self.context['planning'] = planning_data
                st.success("✅ Planning stage completed")
                return planning_data
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            st.error(f"Planning stage failed: {str(e)}")
            raise

    def content_generation_stage(self, planning_data: Dict, topic: str, level: str, format: str) -> Dict:
        """Stage 2: Content Generation - Create study materials"""
        prompt = f"""Based on this study plan, generate comprehensive study content:
        {json.dumps(planning_data, indent=2)}

        Topic: {topic}
        Level: {level}
        Format: {format}

        Generate detailed content with:
        1. Introduction to the topic
        2. Main content sections with explanations, examples, and key points
        3. Summary of key concepts
        4. References or additional resources
        5. Visual descriptions (diagrams, charts where appropriate)

        Make content engaging, clear, and appropriate for the specified level.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.5,
                max_tokens=4096
            )
            
            content = response.choices[0].message.content
            self.context['content'] = content
            
            # Parse into sections
            sections = self._parse_content_into_sections(content)
            self.context['sections'] = sections
            st.success("✅ Content generation stage completed")
            return {'raw_content': content, 'sections': sections}
            
        except Exception as e:
            st.error(f"Content generation failed: {str(e)}")
            raise

    def assessment_stage(self, content: str, topic: str, level: str) -> Dict:
        """Stage 3: Assessment - Create evaluation materials"""
        prompt = f"""Create assessment materials for this content:
        Topic: {topic}
        Level: {level}
        Content: {content[:2000]}... (truncated)

        Generate:
        1. 10 multiple choice questions with answers (4 options each)
        2. 5 short answer questions
        3. 2 discussion questions
        4. Answer key for all questions
        5. Self-assessment rubric
        
        Format as JSON with keys: mcq, short_answer, discussion, answer_key, rubric
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.4,
                max_tokens=3072
            )
            
            result_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                assessment_data = json.loads(json_match.group())
                self.context['assessment'] = assessment_data
                st.success("✅ Assessment stage completed")
                return assessment_data
            else:
                raise ValueError("No valid JSON found in assessment response")
                
        except Exception as e:
            st.error(f"Assessment stage failed: {str(e)}")
            raise

    def review_stage(self, content: Dict, assessment: Dict, topic: str) -> str:
        """Stage 4: Review - Quality check and feedback"""
        prompt = f"""Review the following study materials for quality and completeness:
        Topic: {topic}
        Content Preview: {str(content)[:1000]}...
        Assessment Preview: {str(assessment)[:500]}...

        Provide a comprehensive review covering:
        1. Content accuracy and completeness
        2. Clarity and appropriateness for the level
        3. Assessment quality and alignment with objectives
        4. Suggestions for improvement
        5. Overall rating (1-5 stars)
        6. Specific recommendations for refinement

        Format your response as a structured review.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.3,
                max_tokens=2048
            )
            
            review = response.choices[0].message.content
            self.context['review'] = review
            st.success("✅ Review stage completed")
            return review
            
        except Exception as e:
            st.error(f"Review stage failed: {str(e)}")
            raise

    def refinement_stage(self, content: Dict, assessment: Dict, review: str) -> Dict:
        """Stage 5: Refinement - Polish final version"""
        prompt = f"""Based on this review feedback, refine the study materials:
        Review: {review}
        Original Content: {str(content)[:1500]}...
        Original Assessment: {str(assessment)[:500]}...

        Provide a refined version with:
        1. Improved content addressing feedback
        2. Enhanced assessment materials
        3. Better clarity and structure
        4. Additional examples where needed
        5. A polished final version ready for delivery

        Format as JSON with keys: refined_content, refined_assessment, final_notes
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.3,
                max_tokens=4096
            )
            
            result_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                refined_data = json.loads(json_match.group())
                self.context['refined'] = refined_data
                st.success("✅ Refinement stage completed")
                return refined_data
            else:
                raise ValueError("No valid JSON found in refinement response")
                
        except Exception as e:
            st.error(f"Refinement stage failed: {str(e)}")
            raise

    def _parse_content_into_sections(self, content: str) -> List[Dict]:
        """Helper method to parse content into sections"""
        sections = []
        # Simple section detection based on headings
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            if line.strip().startswith(('#', '##', '###', '**')):
                if current_section:
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(current_content)
                    })
                current_section = line.strip('#* ').strip()
                current_content = []
            else:
                current_content.append(line)
                
        if current_section:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_content)
            })
            
        return sections if sections else [{'title': 'Main Content', 'content': content}]

    def generate_comprehensive_study_pack(self, topic: str, level: str, format: str) -> StudyPack:
        """Generate complete study pack using the workflow"""
        st.write("🚀 Starting AI Study Pack Generation Workflow...")
        
        # Stage 1: Planning
        planning = self.execute_stage("Planning", self.planning_stage, topic, level, format)
        if not planning:
            raise RuntimeError("Planning stage failed")
        
        # Stage 2: Content Generation
        content = self.execute_stage("Content Generation", self.content_generation_stage, 
                                    planning, topic, level, format)
        if not content:
            raise RuntimeError("Content generation stage failed")
        
        # Stage 3: Assessment
        assessment = self.execute_stage("Assessment", self.assessment_stage, 
                                       content['raw_content'], topic, level)
        if not assessment:
            raise RuntimeError("Assessment stage failed")
        
        # Stage 4: Review
        review = self.execute_stage("Review", self.review_stage, 
                                   content, assessment, topic)
        if not review:
            raise RuntimeError("Review stage failed")
        
        # Stage 5: Refinement
        refined = self.execute_stage("Refinement", self.refinement_stage, 
                                    content, assessment, review)
        if not refined:
            raise RuntimeError("Refinement stage failed")
        
        # Create study pack
        study_pack = StudyPack(
            topic=topic,
            level=level,
            format=format,
            learning_objectives=planning.get('objectives', []),
            content={
                'original': content['raw_content'],
                'refined': refined.get('refined_content', '')
            },
            assessment={
                'original': assessment,
                'refined': refined.get('refined_assessment', {})
            },
            review_feedback=review,
            final_version=refined.get('final_notes', '')
        )
        
        return study_pack

# ============= UI Components =============
def display_study_pack(pack: StudyPack):
    """Display the generated study pack in a structured format"""
    st.header(f"📚 Study Pack: {pack.topic}")
    st.write(f"**Level:** {pack.level}")
    st.write(f"**Format:** {pack.format}")
    st.write(f"**Created:** {pack.created_at}")
    
    # Learning Objectives
    with st.expander("🎯 Learning Objectives", expanded=True):
        for i, obj in enumerate(pack.learning_objectives, 1):
            st.write(f"{i}. {obj}")
    
    # Content
    with st.expander("📖 Study Content", expanded=True):
        if pack.content.get('refined'):
            st.markdown(pack.content['refined'])
            st.divider()
            with st.expander("View Original Content"):
                st.markdown(pack.content['original'])
        else:
            st.markdown(pack.content.get('original', 'No content available'))
    
    # Assessment
    with st.expander("📝 Assessment Materials", expanded=False):
        if pack.assessment.get('refined'):
            st.json(pack.assessment['refined'])
            st.divider()
            with st.expander("View Original Assessment"):
                st.json(pack.assessment['original'])
        else:
            st.json(pack.assessment.get('original', {}))
    
    # Review Feedback
    with st.expander("📊 Review & Feedback", expanded=False):
        st.markdown(pack.review_feedback or "No review available")
    
    # Final Version
    with st.expander("✨ Final Version", expanded=False):
        st.markdown(pack.final_version or "No final version available")
    
    # Error Log
    if hasattr(pack, 'errors') and pack.errors:
        with st.expander("⚠️ Error Log", expanded=False):
            for error in pack.errors:
                st.warning(f"Stage: {error['stage']} - {error['error']}")

def handle_errors():
    """Display error messages if any"""
    if 'errors' in st.session_state and st.session_state.errors:
        with st.expander("⚠️ Errors Occurred", expanded=False):
            for error in st.session_state.errors:
                st.error(f"Stage: {error['stage']} - {error['error']}")

# ============= Main Application =============
def main():
    st.set_page_config(
        page_title="AI Study Pack Generator",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 AI Study Pack Generator")
    st.markdown("""
    Generate personalized study materials using our multi-stage AI workflow:
    **Planning** → **Content Generation** → **Assessment** → **Review** → **Refinement**
    """)
    
    # Check for API key
    if "GROQ_API_KEY" not in st.secrets:
        st.error("""
        ❌ GROQ_API_KEY not found in Streamlit secrets. 
        
        Please add your GROQ API key to `.streamlit/secrets.toml`:
        ```toml
        GROQ_API_KEY = "your-api-key-here"
