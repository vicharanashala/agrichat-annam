"""
Agricultural AI Tasks for CrewAI-based RAG System.

This module contains task definitions for agricultural knowledge processing workflow,
designed to ensure high-quality responses for Indian agricultural queries.
"""

from crewai import Task
from typing import List, Dict, Optional


class AgriTasksManager:
    """
    Manager class for agricultural AI tasks in the CrewAI workflow.
    
    This class defines and manages the complete task workflow for processing
    agricultural queries with quality assurance and Indian agricultural context.
    
    Features:
    - Structured task definitions for agricultural workflow
    - Quality assurance through grading tasks
    - Indian agricultural context specialization
    - Conversation history integration
    - Fallback handling for complex queries
    """
    
    def __init__(self, agents_manager):
        """
        Initialize agricultural tasks manager.
        
        Args:
            agents_manager: Instance of AgriAgentsManager containing all agents
        """
        self.agents_manager = agents_manager
        self.agents = agents_manager.get_agents()
        
        # Initialize tasks
        self._initialize_tasks()
    
    def _initialize_tasks(self) -> None:
        """Initialize all agricultural processing tasks."""
        self.retriever_task = self._create_retriever_task()
        self.grader_task = self._create_grader_task()
        self.hallucination_task = self._create_hallucination_task()
        self.answer_task = self._create_answer_task()
    
    def _create_retriever_task(self) -> Task:
        """
        Create the main retriever task for agricultural query processing.
        
        This task defines the primary workflow for handling agricultural questions
        with emphasis on Indian agricultural context and systematic tool usage.
        
        Returns:
            Task: Configured retriever task for agricultural queries
        """
        return Task(
            description=(
                "For the Indian agricultural question {question}, you MUST follow this exact sequence for India-specific responses:\n"
                "1. FIRST: Always call the RAG tool to search the vectorstore database for Indian agricultural information\n"
                "   - If conversation_history is provided: {conversation_history}, pass it to the RAG tool for context-aware responses about Indian farming\n"
                "   - This enables chain of thought processing for follow-up queries about Indian agriculture\n"
                "   - Ensure all responses focus on Indian crop varieties, Indian soil conditions, and Indian farming practices\n"
                "2. ONLY IF the RAG tool returns '__FALLBACK__' or 'I don't have enough information': then call the fallback tool for Indian agricultural knowledge\n"
                "3. Return the answer with Indian agricultural context and the correct source label\n"
                "CRITICAL: You must attempt the RAG tool first before any other tool. Do not skip this step.\n"
                "IMPORTANT: Always include conversation_history when calling the RAG tool, even if it's an empty list.\n"
                "ESSENTIAL: All responses must be specific to Indian agriculture, Indian regions, and Indian farming conditions."
            ),
            expected_output=(
                "The answer from the appropriate tool focused on Indian agriculture:\n"
                "- RAG database if Indian agricultural information is available\n"
                "- Fallback LLM with Indian agricultural knowledge if RAG returns '__FALLBACK__'\n"
                "Keep the response concise, structured, user-friendly, and specifically applicable to Indian farming conditions without technical source labels.\n"
                "Ensure all advice is relevant to Indian soil types, climate patterns, crop varieties, and regional farming practices."
            ),
            agent=self.agents['retriever']
        )
    
    def _create_grader_task(self) -> Task:
        """
        Create the grader task for response relevance assessment.
        
        This task evaluates whether retrieved content is relevant to the agricultural
        question and applicable to Indian farming conditions.
        
        Returns:
            Task: Configured grader task for relevance assessment
        """
        return Task(
            description=(
                "Based on the response from the retriever task for the Indian agricultural question {question}, "
                "evaluate whether the retrieved content is relevant to the question and applicable to Indian farming conditions. "
                "Consider regional variations, crop-specific information, and seasonal factors relevant to Indian agriculture."
            ),
            expected_output=(
                "Binary score 'yes' or 'no' to indicate if the retrieved answer is relevant to the question and applicable to Indian agriculture. "
                "Answer 'yes' only if it meaningfully addresses the question with Indian agricultural context. "
                "Answer 'no' if irrelevant, off-topic, or not applicable to Indian farming conditions. "
                "Reply only 'yes' or 'no' without any explanation or preamble."
            ),
            agent=self.agents['grader'],
            context=[self.retriever_task],
        )
    
    def _create_hallucination_task(self) -> Task:
        """
        Create the hallucination assessment task for fact verification.
        
        This task assesses whether answers are grounded in Indian agricultural
        facts and supported by evidence applicable to Indian farming conditions.
        
        Returns:
            Task: Configured hallucination assessment task
        """
        return Task(
            description=(
                "Based on the graded response for the Indian agricultural question {question}, "
                "assess if the answer is grounded in Indian agricultural facts and supported by evidence applicable to Indian farming conditions. "
                "Verify that agricultural practices, crop recommendations, and farming techniques are suitable for Indian climate, soil, and regional conditions."
            ),
            expected_output=(
                "Binary score 'yes' or 'no' to indicate if the answer is factually sound for Indian agriculture and aligned with the question. "
                "Answer 'yes' if the answer is useful, factually supported, and applicable to Indian farming conditions; 'no' otherwise. "
                "Consider accuracy of crop varieties, seasonal timing, regional applicability, and farming practice validity for Indian conditions. "
                "Reply only 'yes' or 'no' without any explanation."
            ),
            agent=self.agents['hallucination_grader'],
            context=[self.grader_task],
        )
    
    def _create_answer_task(self) -> Task:
        """
        Create the final answer task for response generation or fallback.
        
        This task decides whether to return the processed answer or perform
        a fallback search for additional Indian agricultural information.
        
        Returns:
            Task: Configured answer task for final response generation
        """
        return Task(
            description=(
                "Based on the hallucination grading for the Indian agricultural question {question}, "
                "decide whether to return the answer or perform a fallback search for Indian agricultural information. "
                "If grading is 'yes', return a clear, concise answer focused on Indian agricultural context. "
                "If grading is 'no', perform a search using the fallback tool for Indian agricultural knowledge and return India-specific information. "
                "If unable to produce a valid answer about Indian agriculture, respond with 'Sorry! unable to find a valid response for Indian agricultural conditions'. "
                "Ensure all recommendations are practical and applicable to Indian farming systems."
            ),
            expected_output=(
                "Return a clear and concise answer focused on Indian agriculture if hallucination task graded 'yes'. "
                "If 'no', invoke the fallback tool's search for Indian agricultural information and return the India-specific answer. "
                "All responses must be applicable to Indian farming conditions, Indian crop varieties, and Indian regional agriculture. "
                "Include practical implementation advice suitable for Indian farming systems and economic conditions. "
                "Otherwise, respond with 'Sorry! unable to find a valid response for Indian agricultural conditions'."
            ),
            agent=self.agents['answer_grader'],
            context=[self.hallucination_task],
        )
    
    def get_tasks(self) -> List[Task]:
        """
        Get all configured tasks in execution order.
        
        Returns:
            List[Task]: List of tasks in proper execution sequence
        """
        return [
            self.retriever_task,
            self.grader_task,
            self.hallucination_task,
            self.answer_task
        ]
    
    def get_task_by_name(self, task_name: str) -> Optional[Task]:
        """
        Get a specific task by name.
        
        Args:
            task_name: Name of the task to retrieve
            
        Returns:
            Optional[Task]: Task instance or None if not found
        """
        task_mapping = {
            'retriever': self.retriever_task,
            'grader': self.grader_task,
            'hallucination': self.hallucination_task,
            'answer': self.answer_task
        }
        
        return task_mapping.get(task_name)
    
    def get_workflow_description(self) -> Dict[str, str]:
        """
        Get description of the complete workflow.
        
        Returns:
            Dict[str, str]: Description of each task in the workflow
        """
        return {
            'retriever': 'Primary agricultural query processing with RAG and fallback tools',
            'grader': 'Relevance assessment for Indian agricultural context',
            'hallucination': 'Fact verification for Indian agricultural accuracy',
            'answer': 'Final response generation with quality assurance'
        }
    
    def validate_workflow(self) -> Dict[str, bool]:
        """
        Validate the complete task workflow configuration.
        
        Returns:
            Dict[str, bool]: Validation status for each component
        """
        validation_results = {}
        
        try:
            # Check task existence
            validation_results['retriever_task'] = hasattr(self.retriever_task, 'description')
            validation_results['grader_task'] = hasattr(self.grader_task, 'description')
            validation_results['hallucination_task'] = hasattr(self.hallucination_task, 'description')
            validation_results['answer_task'] = hasattr(self.answer_task, 'description')
            
            # Check task dependencies
            validation_results['grader_context'] = bool(self.grader_task.context)
            validation_results['hallucination_context'] = bool(self.hallucination_task.context)
            validation_results['answer_context'] = bool(self.answer_task.context)
            
            # Check agent assignments
            validation_results['retriever_agent'] = hasattr(self.retriever_task.agent, 'role')
            validation_results['grader_agent'] = hasattr(self.grader_task.agent, 'role')
            validation_results['hallucination_agent'] = hasattr(self.hallucination_task.agent, 'role')
            validation_results['answer_agent'] = hasattr(self.answer_task.agent, 'role')
            
        except Exception as e:
            validation_results['error'] = str(e)
        
        return validation_results


# Legacy task exports for backward compatibility
def create_legacy_tasks(agents_manager):
    """
    Create legacy task instances for backward compatibility.
    
    Args:
        agents_manager: Instance of AgriAgentsManager
        
    Returns:
        Tuple of task instances
    """
    tasks_manager = AgriTasksManager(agents_manager)
    
    return (
        tasks_manager.retriever_task,
        tasks_manager.grader_task,
        tasks_manager.hallucination_task,
        tasks_manager.answer_task
    )


if __name__ == "__main__":
    # Test the tasks manager
    from agents.crew_agents import get_agents_manager
    
    agents_manager = get_agents_manager()
    tasks_manager = AgriTasksManager(agents_manager)
    
    # Validate workflow
    validation = tasks_manager.validate_workflow()
    print(f"Workflow Validation: {validation}")
    
    # Get workflow description
    workflow_desc = tasks_manager.get_workflow_description()
    print(f"Workflow Description: {workflow_desc}")
